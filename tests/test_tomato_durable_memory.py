from dataclasses import asdict
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import numpy as np
import pytest
from tools.tomato_durable_memory import DurableTaskMemoryShadow,select_named_positions
from tools.tomato_task_memory import Observation

NAMES=[f'{side}_fr3_joint{i}.position' for side in ['left','right'] for i in range(7)]+[f'hand{i}.position' for i in range(44)]
IDENTITY=dict(session_id='s',episode_id='episode',model_version='b0',model_sha256='a'*64,manifest_sha256='b'*64,joint_names=NAMES)


def observation(i,k=0,age=.04):
    p=np.full(4,.01);p[k]=.97
    return asdict(Observation('s',i,f's:{i}',i*.5,i*.5+age,tuple(p),tuple(p),'b0'))


def fact(i=2,**updates):
    return dict(dict(session_id='s',model_version='b0',decision_id='decision1',from_phase=0,to_phase=1,prompt_version=1,observation_id=f's:{i}',observation_timestamp=i*.5,decision_timestamp=i*.5+.04,arrival_timestamp=i*.5+.04,source='controller_current_confirmation'),**updates)


def test_durable_restart_preserves_pending_and_confirmed(tmp_path):
    p=tmp_path/'memory.db'
    m=DurableTaskMemoryShadow(p,IDENTITY);m.submit('o0','observation',observation(0));m.submit('o1','observation',observation(1,1));before=m.view(now=.54);m.close()
    m=DurableTaskMemoryShadow(p,IDENTITY);assert m.view(now=.54)==before
    assert not m.view()['perception_is_fresh']
    m.submit('d1','controller_fact',fact());m.submit('o2','observation',observation(2,1));v=m.view(now=1.04);assert v['controller_reported_phase']==v['perception']['confirmed_phase']==1
    m.close();m=DurableTaskMemoryShadow(p,IDENTITY);assert m.view(now=1.04)==v;m.close()


def test_stale_observation_cannot_delete_delayed_controller_fact(tmp_path):
    with DurableTaskMemoryShadow(tmp_path/'memory.db',IDENTITY) as m:
        m.submit('o0','observation',observation(0))
        r=m.submit('d1','controller_fact',fact(arrival_timestamp=2.5));assert r['accepted']
        r=m.submit('late','observation',observation(2,1,age=1.5));assert not r['accepted'] and r['reason']=='stale_or_future_dated'
        v=m.view(now=2.5);assert v['controller_reported_phase']==1 and len(v['controller_facts'])==1
        assert not v['perception_is_fresh'] and not v['suggests_prompt_update']
        assert v['controller_perception_disagree']


def test_idempotency_has_no_duplicate_transition_after_restart(tmp_path):
    p=tmp_path/'memory.db';m=DurableTaskMemoryShadow(p,IDENTITY);first=m.submit('d1','controller_fact',fact());m.close()
    with DurableTaskMemoryShadow(p,IDENTITY) as m:
        ordinal=m.ordinal;assert m.submit('d1','controller_fact',fact())==first and m.ordinal==ordinal
        with pytest.raises(ValueError,match='Idempotency'):m.submit('d1','controller_fact',fact(to_phase=2))
        assert len(m.facts)==1


def test_future_and_skipped_stages_are_not_controller_facts(tmp_path):
    with DurableTaskMemoryShadow(tmp_path/'memory.db',IDENTITY) as m:
        assert not m.submit('future','controller_fact',fact(source='future_warning'))['accepted']
        assert not m.submit('skip','controller_fact',fact(to_phase=4))['accepted']
        assert m.controller_phase==0
        assert m.submit('d','controller_fact',fact())['accepted']
        assert not m.submit('repeated','controller_fact',fact())['accepted']
        assert not m.submit('old','controller_fact',fact(session_id='other'))['accepted']


def test_identity_change_never_reinitializes_existing_memory(tmp_path):
    p=tmp_path/'memory.db';DurableTaskMemoryShadow(p,IDENTITY).close()
    with pytest.raises(ValueError,match='identity/code/schema'):DurableTaskMemoryShadow(p,dict(IDENTITY,model_version='other'))
    DurableTaskMemoryShadow(p,IDENTITY).close()


def test_poll_after_restore_expires_old_candidates_without_advancing(tmp_path):
    p=tmp_path/'memory.db';m=DurableTaskMemoryShadow(p,IDENTITY);m.submit('o1','observation',observation(1,1));assert m.memory.pending;m.close()
    with DurableTaskMemoryShadow(p,IDENTITY) as m:
        assert m.submit('tick','poll',dict(now=3.))['accepted']
        assert m.memory.pending is None and m.controller_phase==0 and not m.view(now=3.)['perception_is_fresh']


def test_single_writer_and_corruption_detection(tmp_path):
    p=tmp_path/'memory.db';m=DurableTaskMemoryShadow(p,IDENTITY)
    with pytest.raises(RuntimeError,match='active writer'):DurableTaskMemoryShadow(p,IDENTITY)
    m.submit('o0','observation',observation(0));m.close()
    with sqlite3.connect(p) as conn:conn.execute("UPDATE commands SET result='{}'")
    with pytest.raises(ValueError,match='corruption'):DurableTaskMemoryShadow(p,IDENTITY)


def test_missing_committed_tail_is_detected(tmp_path):
    p=tmp_path/'memory.db';m=DurableTaskMemoryShadow(p,IDENTITY);m.submit('o0','observation',observation(0));m.close()
    with sqlite3.connect(p) as conn:conn.execute('DELETE FROM commands')
    with pytest.raises(ValueError,match='tail mismatch'):DurableTaskMemoryShadow(p,IDENTITY)


def test_actual_process_exit_after_commit_retries_once(tmp_path):
    p=tmp_path/'memory.db';code='''
import os,json,sys
from tools.tomato_durable_memory import DurableTaskMemoryShadow
m=DurableTaskMemoryShadow(sys.argv[1],json.loads(sys.argv[2]))
m.submit('d1','controller_fact',json.loads(sys.argv[3]))
os._exit(73)
'''
    child=subprocess.run([sys.executable,'-c',code,str(p),json.dumps(IDENTITY),json.dumps(fact())]);assert child.returncode==73
    with DurableTaskMemoryShadow(p,IDENTITY) as m:
        assert m.controller_phase==1 and m.ordinal==1
        m.submit('d1','controller_fact',fact());assert m.ordinal==1 and len(m.facts)==1
        assert m.conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'


def test_named116_to58_reordered_positions_not_velocities():
    source=[n.replace('.position','.velocity') for n in NAMES]+NAMES[::-1]
    raw=np.arange(2*116,dtype=float).reshape(2,116);q=select_named_positions(raw,source,NAMES)
    np.testing.assert_array_equal(q,raw[:,np.arange(115,57,-1)])
    with pytest.raises(ValueError,match='Missing'):select_named_positions(raw,[n.replace('.position','.action') for n in source],NAMES)
    with pytest.raises(ValueError,match='Unambiguous'):select_named_positions(raw,source[:-1]+[source[0]],NAMES)
    q[0,0]=-1;assert raw[0,115]==115


def test_process_exit_before_commit_does_not_publish_fact(tmp_path):
    p=tmp_path/'memory.db';code='''
import os,json,sys
from tools.tomato_durable_memory import DurableTaskMemoryShadow
m=DurableTaskMemoryShadow(sys.argv[1],json.loads(sys.argv[2]))
original=m._apply
def interrupted(command):
    original(command)
    os._exit(74)
m._apply=interrupted
m.submit('d1','controller_fact',json.loads(sys.argv[3]))
'''
    child=subprocess.run([sys.executable,'-c',code,str(p),json.dumps(IDENTITY),json.dumps(fact())]);assert child.returncode==74
    with DurableTaskMemoryShadow(p,IDENTITY) as m:
        assert m.controller_phase==0 and m.ordinal==0
        assert m.submit('d1','controller_fact',fact())['accepted'] and m.controller_phase==1


def test_snapshot_is_detached_and_invalid_clock_cannot_refresh(tmp_path):
    with DurableTaskMemoryShadow(tmp_path/'memory.db',IDENTITY) as m:
        m.submit('d','controller_fact',fact());v=m.view(now=1.04);v['controller_facts'][0]['to_phase']=6
        assert m.view(now=1.04)['controller_facts'][0]['to_phase']==1
        assert not m.view(now=-1)['perception_is_fresh']
        assert not m.submit('oldclock','poll',dict(now=.5))['accepted']
        assert m.controller_phase==1


def test_compact_context_is_bounded_shadow_and_preserves_prefix(tmp_path):
    from tools.tomato_memory_context import compact_context
    with DurableTaskMemoryShadow(tmp_path/'memory.db',IDENTITY) as m:
        m.submit('d','controller_fact',fact())
        for i in range(3,12):m.submit(f'o{i}','observation',observation(i,1))
        c=compact_context(m,now=5.54,recent_limit=4)
        assert len(c['recent_recognition_history'])==4 and len(c['ordered_decision_history'])==1
        assert c['fresh_current'] is not None and not c['suggests_prompt_update']
        assert len(json.dumps(c).encode())<8192
        c['ordered_decision_history'][0]['to_phase']=6
        assert m.controller_phase==1
        with pytest.raises(ValueError,match='nondecreasing'):compact_context(m,now=1.)
        old=compact_context(m,now=8.);assert old['fresh_current'] is None and len(old['ordered_decision_history'])==1
        no_recent=compact_context(m,now=8.,recent_limit=0);assert not no_recent['recent_recognition_history']
        with pytest.raises(ValueError,match='Budget'):compact_context(m,now=8.,max_bytes=256)


def test_existing_unrelated_database_is_not_modified(tmp_path):
    p=tmp_path/'other.db'
    with sqlite3.connect(p) as conn:conn.execute('CREATE TABLE important (value TEXT)');conn.execute("INSERT INTO important VALUES('keep')")
    with pytest.raises(ValueError,match='unrelated'):DurableTaskMemoryShadow(p,IDENTITY)
    with sqlite3.connect(p) as conn:
        assert conn.execute('SELECT value FROM important').fetchone()==('keep',)
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()==[('important',)]


def test_failed_transaction_does_not_leave_uncommitted_in_memory_fact(tmp_path):
    with DurableTaskMemoryShadow(tmp_path/'memory.db',IDENTITY) as m:
        m.conn.execute("CREATE TRIGGER fail_commands BEFORE INSERT ON commands BEGIN SELECT RAISE(FAIL,'write failure fixture'); END")
        m.conn.commit()
        with pytest.raises(sqlite3.DatabaseError):m.submit('d','controller_fact',fact())
        assert m.controller_phase==0 and not m.facts and m.ordinal==0
        m.conn.execute('DROP TRIGGER fail_commands');m.conn.commit()
        assert m.submit('d','controller_fact',fact())['accepted'] and m.controller_phase==1
