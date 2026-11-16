import pytest
from tools.tomato_review_ui.handshake_improvement_study import ResearchHandshake,VARIANTS,run
from tools.tomato_review_ui.handshake import Evidence,LABELS

def ev(t,current=0,future=1):
    def p(i):return tuple(.97 if k==i else .01 for k in range(4))
    return Evidence(t,t+.09,p(current),p(future))

def test_late_ack_is_research_only_and_preserves_active():
    h=ResearchHandshake(VARIANTS['L_lateACK1']);h.prediction(ev(0));h.planning(.5);h.timer(2.)
    assert h.state=='RECOVERY_REQUIRED';h.prediction(ev(2.,current=1,future=1))
    assert h.state=='READY' and h.active=='harvest' and h.version==1
    assert any(e['kind']=='late_ack' for e in h.events)

def test_two_distinct_late_observations_and_hard_limit():
    h=ResearchHandshake(VARIANTS['S_lateACK2']);h.prediction(ev(0));h.planning(.5);h.timer(2.)
    h.prediction(ev(2.,current=1,future=1));assert h.state=='RECOVERY_REQUIRED'
    with pytest.raises(ValueError):h.prediction(ev(2.,current=1,future=1))
    h.prediction(ev(2.5,current=1,future=1));assert h.state=='READY'
    x=ResearchHandshake(VARIANTS['L_lateACK1']);x.prediction(ev(0));x.planning(.5);x.timer(3.5);x.prediction(ev(4,current=1,future=1));assert x.state=='RECOVERY_REQUIRED'

def test_contract_count_uses_commits_not_gt():
    h=ResearchHandshake(VARIANTS['C_lateACK2_contract2cycles']);h.active='return';h.state='READY'
    h.prediction(ev(0,current=2,future=3));h.planning(.5);assert h.active=='adjust' and h.continuations==1
    h.state='READY';h.active='return';h.prediction(ev(1,current=2,future=3));h.planning(1.5)
    assert h.active=='return' and h.version==1

def test_failclosed_blocks_first_return_also():
    h=ResearchHandshake(VARIANTS['F_lateACK2_failclosed']);h.active='return';h.prediction(ev(0,current=2,future=3));h.planning(.5)
    assert h.version==0 and any(e['kind']=='continuation_required' for e in h.events)
