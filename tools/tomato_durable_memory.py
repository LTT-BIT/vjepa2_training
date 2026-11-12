"""Transactional, replayable TaskMemory SHADOW. No Prompt/action write interface.

A controller-confirmed fact is historical evidence about a decision, NOT fresh
perception or physical completion. The original frozen TaskMemory is untouched.
"""
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from .tomato_task_memory import TaskMemory,Observation
from . import tomato_dataset as ds


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def select_named_positions(values,source_names,target_names):
    """Select measured positions by exact names; never infer from 116D layout."""
    import numpy as np
    names=list(source_names);target=list(target_names);q=np.asarray(values)
    if len(set(names))!=len(names) or len(target)!=58 or len(set(target))!=58:
        raise ValueError('Unambiguous named source and 58D target required')
    if q.ndim<1 or q.shape[-1]!=len(names) or not np.isfinite(q).all():
        raise ValueError('Finite state matching the declared named schema required')
    if sum('fr3' in n for n in target)!=14 or any(not n.endswith('.position') for n in target):
        raise ValueError('Named measured 14-arm/44-hand positions required')
    try:columns=[names.index(n) for n in target]
    except ValueError as exc:raise ValueError('Missing exact position name; no dimension-only fallback') from exc
    return q[...,columns].copy()


class DurableTaskMemoryShadow:
    protocol='tomato-durable-memory-shadow-v1'

    def __init__(self,path,identity,*,short_size=16,long_size=256):
        required={'session_id','episode_id','model_version','model_sha256','manifest_sha256','joint_names'}
        if set(identity)!=required or not all(isinstance(identity[k],str) and identity[k] for k in required-{'joint_names'}):
            raise ValueError('Exact session, episode, model and frozen-data identities required')
        for key in ['model_sha256','manifest_sha256']:
            if len(identity[key])!=64 or any(c not in '0123456789abcdef' for c in identity[key]):raise ValueError('SHA256 identity required')
        self.identity=json.loads(canonical(identity));self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.binding=dict(protocol=self.protocol,identity=self.identity,short_size=short_size,long_size=long_size,task_memory_sha256=ds.digest(Path(__file__).with_name('tomato_task_memory.py')),adapter_sha256=ds.digest(__file__),max_observation_age_s=1.,mode='M1_shadow_only')
        self.handle=self.path.with_suffix(self.path.suffix+'.writer.lock').open('a')
        try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close();raise RuntimeError('An active writer already owns this Memory session')
        self.conn=None
        try:
            existed=self.path.exists()
            self.conn=sqlite3.connect(self.path)
            if existed:
                tables={r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {'binding','commands','tip'}<=tables:raise ValueError('Refuse unrelated or incomplete existing database')
                bound=self.conn.execute('SELECT body FROM binding WHERE singleton=1').fetchone()
                if bound is None or bound[0]!=canonical(self.binding):raise ValueError('Memory identity/code/schema mismatch; never reset an existing session')
            self.conn.execute('PRAGMA journal_mode=WAL');self.conn.execute('PRAGMA synchronous=FULL')
            self.conn.execute('CREATE TABLE IF NOT EXISTS binding (singleton INTEGER PRIMARY KEY CHECK(singleton=1), body TEXT NOT NULL)')
            self.conn.execute('CREATE TABLE IF NOT EXISTS tip (singleton INTEGER PRIMARY KEY CHECK(singleton=1), ordinal INTEGER NOT NULL, sha256 TEXT NOT NULL)')
            self.conn.execute('CREATE TABLE IF NOT EXISTS commands (ordinal INTEGER PRIMARY KEY, command_id TEXT UNIQUE NOT NULL, body TEXT NOT NULL, result TEXT NOT NULL, previous_sha256 TEXT NOT NULL, sha256 TEXT NOT NULL)')
            stored=self.conn.execute('SELECT body FROM binding WHERE singleton=1').fetchone()
            if stored is None:
                if self.conn.execute('SELECT COUNT(*) FROM commands').fetchone()[0]:raise ValueError('Journal without identity binding')
                self.conn.execute('INSERT INTO binding VALUES(1,?)',(canonical(self.binding),));self.conn.execute('INSERT INTO tip VALUES(1,0,?)',(ds.fingerprint(self.binding),));self.conn.commit()
            elif stored[0]!=canonical(self.binding):raise ValueError('Memory identity/code/schema mismatch; never reset an existing session')
            self._rebuild()
        except BaseException:
            self.close();raise

    def _initial(self):
        i=self.identity
        self.memory=TaskMemory(i['session_id'],i['model_version'],mode='M1',short_size=self.binding['short_size'],long_size=self.binding['long_size'],joint_names=i['joint_names'])
        self.controller_phase=0;self.prompt_version=0;self.facts=[];self.arrival_clock=None;self.ordinal=0;self.tail=ds.fingerprint(self.binding)

    def _rebuild(self):
        self._initial()
        for ordinal,cid,body,result,previous,sha in self.conn.execute('SELECT ordinal,command_id,body,result,previous_sha256,sha256 FROM commands ORDER BY ordinal'):
            if ordinal!=self.ordinal+1 or previous!=self.tail or sha!=self._hash(ordinal,cid,body,result,previous):raise ValueError('Memory journal chain/sequence corruption')
            actual=canonical(self._apply(json.loads(body)))
            if actual!=result:raise ValueError('Memory replay differs from committed result; refuse silent recovery')
            self.ordinal=ordinal;self.tail=sha
        if self.conn.execute('SELECT ordinal,sha256 FROM tip WHERE singleton=1').fetchone()!=(self.ordinal,self.tail):raise ValueError('Memory journal committed tail mismatch')

    @staticmethod
    def _hash(ordinal,cid,body,result,previous):
        return hashlib.sha256(canonical([ordinal,cid,body,result,previous]).encode()).hexdigest()

    def _apply(self,command):
        kind=command.get('kind');payload=command.get('payload',{});accepted=False;reason='unknown_command'
        if not isinstance(payload,dict):return self._receipt(False,'invalid_payload')
        try:
            if kind=='observation':
                if set(payload)-set(Observation.__dataclass_fields__):return self._receipt(False,'unexpected_observation_fields')
                o=Observation(**payload)
                # Cross-session packets must not expire or advance the current session.
                if o.session_id!=self.identity['session_id'] or o.model_version!=self.identity['model_version']:return self._receipt(False,'identity_mismatch')
                if self.arrival_clock is not None and o.arrival_timestamp<self.arrival_clock:return self._receipt(False,'arrival_clock_reversal')
                r=self.memory.observe(o,b0_phase=self.controller_phase);accepted=r['valid'];reason='fresh_observation_recorded' if accepted else (self.memory.short[-1]['reason'] if self.memory.short else 'invalid_observation')
                if accepted:self.arrival_clock=float(o.arrival_timestamp)
            elif kind=='controller_fact':
                required={'session_id','model_version','decision_id','from_phase','to_phase','prompt_version','observation_id','observation_timestamp','decision_timestamp','arrival_timestamp','source'}
                if set(payload)!=required:return self._receipt(False,'invalid_controller_fact_schema')
                if (payload['session_id'],payload['model_version'])!=(self.identity['session_id'],self.identity['model_version']):return self._receipt(False,'identity_mismatch')
                stamps=[payload[k] for k in ['observation_timestamp','decision_timestamp','arrival_timestamp']]
                if not all(isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) for t in stamps) or not 0<=stamps[0]<=stamps[1]<=stamps[2]:return self._receipt(False,'invalid_fact_times')
                if self.arrival_clock is not None and stamps[2]<self.arrival_clock:return self._receipt(False,'arrival_clock_reversal')
                if payload['source']!='controller_current_confirmation':return self._receipt(False,'future_prediction_is_not_controller_confirmation')
                if not isinstance(payload['decision_id'],str) or not payload['decision_id'] or any(f['decision_id']==payload['decision_id'] for f in self.facts):return self._receipt(False,'duplicate_or_invalid_decision_id')
                oid=payload['observation_id'];prefix=self.identity['session_id']+':'
                if not isinstance(oid,str) or not oid.startswith(prefix) or not oid[len(prefix):].isdigit():return self._receipt(False,'invalid_support_observation_id')
                if any(type(payload[k]) is not int for k in ['from_phase','to_phase','prompt_version']) or payload['from_phase']!=self.controller_phase or payload['to_phase']!=self.controller_phase+1 or payload['to_phase']>6 or payload['prompt_version']!=self.prompt_version+1:return self._receipt(False,'nonadjacent_or_obsolete_controller_fact')
                if self.facts and stamps[1]<self.facts[-1]['decision_timestamp']:return self._receipt(False,'controller_decision_time_reversal')
                self.facts.append(deepcopy(payload));self.controller_phase=payload['to_phase'];self.prompt_version=payload['prompt_version'];self.arrival_clock=float(stamps[2]);accepted=True;reason='historical_controller_fact_recorded_not_fresh_perception'
            elif kind=='poll':
                if set(payload)!={'now'}:return self._receipt(False,'invalid_poll_schema')
                now=float(payload['now'])
                if not math.isfinite(now) or now<0 or self.arrival_clock is not None and now<self.arrival_clock:return self._receipt(False,'arrival_clock_reversal')
                accepted=self.memory.poll(now);self.arrival_clock=now;reason='clock_polled_no_stage_change'
        except (ValueError,TypeError,AttributeError,KeyError,OverflowError):
            # Invalid envelopes are persisted as rejections, not promoted to facts.
            accepted=False;reason='invalid_command_envelope'
        return self._receipt(accepted,reason)

    def _receipt(self,accepted,reason):
        return dict(accepted=bool(accepted),reason=reason,controller_reported_phase=self.controller_phase,controller_prompt_version=self.prompt_version,perception_recorded_phase=self.memory.phase,perception_candidate_phase=self.memory.pending.to_phase if self.memory.pending else None,memory_version=self.memory.version,confirmed_controller_facts=len(self.facts),suggests_prompt_update=False,physical_success='unknown')

    def submit(self,command_id,kind,payload):
        if not isinstance(command_id,str) or not command_id:raise ValueError('Stable idempotency command ID required')
        body=canonical(dict(kind=kind,payload=payload))
        try:
            self.conn.execute('BEGIN IMMEDIATE')
            prior=self.conn.execute('SELECT body,result FROM commands WHERE command_id=?',(command_id,)).fetchone()
            if prior:
                self.conn.rollback()
                if prior[0]!=body:raise ValueError('Idempotency key reused with different command')
                return json.loads(prior[1])
            result=canonical(self._apply(json.loads(body)));ordinal=self.ordinal+1;sha=self._hash(ordinal,command_id,body,result,self.tail)
            self.conn.execute('INSERT INTO commands VALUES(?,?,?,?,?,?)',(ordinal,command_id,body,result,self.tail,sha));self.conn.execute('UPDATE tip SET ordinal=?,sha256=? WHERE singleton=1',(ordinal,sha));self.conn.commit();self.ordinal=ordinal;self.tail=sha
            return json.loads(result)
        except BaseException:
            self.conn.rollback();self._rebuild();raise

    def view(self,*,now=None):
        # After process restart callers must provide a current clock in the same
        # time domain; loading historical rows cannot make them fresh again.
        fresh=False
        if isinstance(now,(int,float)) and math.isfinite(now) and self.memory.last_valid is not None and (self.arrival_clock is None or now>=self.arrival_clock):fresh=0<=now-self.memory.last_valid<=1.
        return deepcopy(dict(protocol=self.protocol,identity=self.identity,controller_reported_phase=self.controller_phase,controller_prompt_version=self.prompt_version,controller_facts=self.facts,perception=self.memory.snapshot(),perception_is_fresh=bool(fresh),controller_perception_disagree=self.controller_phase!=self.memory.phase,journal_ordinal=self.ordinal,journal_sha256=self.tail,mode='shadow_only',suggests_prompt_update=False,physical_success='unknown',learned_memory=False))

    def close(self):
        if self.conn is not None:self.conn.close();self.conn=None
        if not self.handle.closed:fcntl.flock(self.handle,fcntl.LOCK_UN);self.handle.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
