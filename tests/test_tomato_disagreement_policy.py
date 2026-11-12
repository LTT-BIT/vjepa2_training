import numpy as np
import pytest
from tools.tomato_disagreement_policy import DisagreementPolicy, ARMS
from tools.tomato_state_recovery import B0State


def p(k):
    v=np.full(4,.01);v[k]=.97
    return v


@pytest.mark.parametrize('arm',ARMS)
def test_future_alone_never_changes_prompt(arm):
    m=DisagreementPolicy(arm)
    for t in np.arange(0,4,.5):
        r=m.update(p(0),p(1),t)
        assert r['phase']==0 and r['prompt_version']==0


def test_advisory_matches_b0_on_fresh_inputs():
    rng=np.random.default_rng(1);b=B0State();a=DisagreementPolicy('advisory')
    for i in range(1000):
        c,f=p(int(rng.integers(4))),p(int(rng.integers(4)))
        x=b.update(c,f,i*.5);y=a.update(c,f,i*.5)
        assert (x['phase'],x['warning'])==(y['phase'],y['warning'])


def test_conflict_cannot_veto_current_forever():
    a=DisagreementPolicy('bounded_conflict')
    r=[a.update(p(1),p(0),i*.5) for i in range(3)]
    assert [x['phase'] for x in r]==[0,0,1]
    assert r[1]['reason']=='bounded_future_conflict_wait_one_fresh_current'


def test_valid_lookahead_does_not_delay_short_stage():
    a=DisagreementPolicy('bounded_conflict')
    a.update(p(1),p(2),0)
    assert a.update(p(1),p(2),.5)['phase']==1


@pytest.mark.parametrize('bad', ['gap','duplicate','stale','nan'])
def test_invalid_and_gap_break_confirmation(bad):
    a=DisagreementPolicy('advisory');a.update(p(1),p(1),0)
    if bad=='gap':
        assert a.update(p(1),p(1),1)['phase']==0
    else:
        r=a.update(p(1),p(1),0 if bad=='duplicate' else .5,
                   now=2 if bad=='stale' else None) if bad!='nan' else a.update([np.nan]*4,p(1),.5)
        assert not r['valid']
        assert a.update(p(1),p(1),1)['phase']==0


def test_terminal_and_repeated_classes_cannot_skip():
    a=DisagreementPolicy('bounded_conflict');t=0
    for k in [1,2,3,0,1,2]:
        for _ in range(3):a.update(p(k),p(k),t);t+=.5
    assert a.phase==6
    for _ in range(4):a.update(p(3),p(3),t);t+=.5
    assert a.phase==6 and a.version==6


def test_prefix_causal_and_episode_isolated():
    c=[0,1,1,2,2,3,3,0,0];f=[1,1,1,0,0,3,0,0,0]
    def run(n):
        a=DisagreementPolicy('bounded_conflict')
        return [a.update(p(c[i]),p(f[i]),i*.5) for i in range(n)]
    full=run(len(c))
    for n in range(1,len(c)+1):assert run(n)==full[:n]
    assert DisagreementPolicy('bounded_conflict').phase==0


def test_preparation_cancel_does_not_change_prompt():
    a=DisagreementPolicy('advisory');a.update(p(0),p(1),0);a.update(p(0),p(0),.5)
    assert [e['kind'] for e in a.events]==['prepare','cancel_preparation']
    assert a.phase==a.version==0


@pytest.mark.parametrize('arm', ARMS)
def test_low_current_or_illegal_next_does_not_skip(arm):
    a=DisagreementPolicy(arm)
    for i in range(5):
        assert a.update(p(2),p(2),i*.5)['phase']==0
    low=np.array([.1,.79,.06,.05])
    for i in range(5,10):
        assert a.update(low,p(1),i*.5)['phase']==0


def test_annotation_or_oracle_argument_is_not_an_input():
    with pytest.raises(TypeError):
        DisagreementPolicy('bounded_conflict').update(p(0),p(1),0,oracle_phase=4)


@pytest.mark.parametrize('arm', ARMS)
def test_all_arms_are_prefix_causal(arm):
    c=[0,1,1,2,2,3,3,0,0]
    f=[1,0,0,2,2,3,0,0,0]
    def run(n):
        a=DisagreementPolicy(arm)
        return [a.update(p(c[i]),p(f[i]),i*.5) for i in range(n)]
    full=run(len(c))
    for n in range(1,len(c)+1):
        assert run(n)==full[:n]
