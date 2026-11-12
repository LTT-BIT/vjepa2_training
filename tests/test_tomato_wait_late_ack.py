"""WAIT_LATE_ACK tests independent of other uncommitted UI features."""
import pytest
from tools.tomato_review_ui.handshake import Handshake,Evidence,LABELS,replay,validate


def m(t,current='harvest',future='harvest',valid=True):
    def p(stage):return tuple(.97 if s==stage else .01 for s in LABELS)
    return Evidence(t,t,p(current),p(future),valid)

def waiting(**extra):
    h=Handshake(dict(late_ack_enabled=True,**extra));h.prediction(m(0,current='pickup'));h.planning(.5);return h


def test_soft_timeout_retains_version_and_requires_two_new_anchors():
    h=waiting();h.timer(2.)
    assert h.state=='WAIT_LATE_ACK' and h.active=='harvest' and h.version==1
    h.prediction(m(2.));assert h.state=='WAIT_LATE_ACK' and h.late_count==1
    with pytest.raises(ValueError):h.prediction(m(2.))
    h.prediction(m(2.5));assert h.state=='READY' and h.version==1
    e=next(e for e in h.events if e['kind']=='late_ack');assert e['ack_delay_s']==2.


def test_regular_ack_still_needs_only_one_post_commit_anchor():
    h=waiting();h.prediction(m(.5));assert h.state=='WAIT_CURRENT_ACK'
    h.prediction(m(1.));assert h.state=='READY'
    assert not any(e['kind']=='late_ack' for e in h.events)


@pytest.mark.parametrize('invalid_kind',['class','confidence','invalid','gap'])
def test_consecutive_late_support_resets(invalid_kind):
    h=waiting();h.timer(2.);h.prediction(m(2.))
    if invalid_kind=='class':h.prediction(m(2.5,current='pickup'))
    elif invalid_kind=='confidence':h.prediction(Evidence(2.5,2.5,(.2,.4,.2,.2),(.01,.97,.01,.01)))
    elif invalid_kind=='invalid':h.prediction(m(2.5,valid=False))
    else:h.prediction(m(3.));assert h.late_count==1;return
    assert h.late_count==0
    h.prediction(m(3.));assert h.state=='WAIT_LATE_ACK'


def test_hard_deadline_total_from_commit_and_never_auto_unlocks():
    h=waiting();assert h.hard_deadline==5.5
    h.timer(2.);h.timer(5.5);assert h.state=='RECOVERY_REQUIRED'
    h.prediction(m(6.));h.prediction(m(6.5));assert h.state=='RECOVERY_REQUIRED' and h.version==1
    h.recover(7.,'operator_authorized');assert h.state=='READY'


def test_new_future_does_not_bypass_wait_and_resets_on_next_commit():
    h=waiting();h.timer(2.);h.prediction(m(2.,future='return'));assert h.pending is None
    h.prediction(m(2.5,future='return'));assert h.state=='SWITCH_REQUESTED'
    assert h.active=='harvest' and h.version==1
    h.planning(2.5);assert h.active=='return' and h.hard_deadline==7.5 and h.late_count==0
    h.timer(5.5);assert h.state=='WAIT_LATE_ACK'  # old deadline is not the new hard deadline


def rows(times):
    return [dict(observed_at=t,heads={h:dict(probabilities=dict(zip(LABELS,getattr(m(t,current='pickup' if t<1 else 'harvest'),h)))) for h in ['current','future']}) for t in times]


def test_timer_without_new_messages_still_reaches_hard_timeout():
    r=replay(rows([0]),dict(late_ack_enabled=True,latency_s=.09),7.)
    assert [(e['kind'],e['time']) for e in r['events'] if e['kind'] in ['timeout','hard_timeout']]==[('timeout',2.),('hard_timeout',5.5)]


def test_exact_hard_deadline_prediction_is_processed_first():
    r=replay(rows([0,4.59,5.09]),dict(late_ack_enabled=True,latency_s=.09),6.)
    assert r['summary']['late_acks']==1 and r['summary']['hard_timeouts']==0
    assert next(e['time'] for e in r['events'] if e['kind']=='late_ack')==5.18
    # Exactly at hard=5.5: the second observation is available then, so it is accepted.
    r=replay(rows([0,4.91,5.41]),dict(late_ack_enabled=True,latency_s=.09),6.)
    assert next(e['time'] for e in r['events'] if e['kind']=='late_ack')==5.5
    assert r['summary']['hard_timeouts']==0


def test_after_hard_deadline_result_is_not_accepted_even_without_timer_call():
    h=waiting();h.prediction(m(5.));h.prediction(m(5.6));assert h.state=='RECOVERY_REQUIRED'


def test_legacy_opt_out_and_config_validation():
    h=Handshake({});h.prediction(m(0,current='pickup'));h.planning(.5);h.timer(2.)
    assert h.state=='RECOVERY_REQUIRED'
    for cfg in [dict(late_ack_enabled='yes'),dict(late_ack_timeout_s=float('nan')),dict(late_ack_enabled=True,ack_timeout_s=2.,late_ack_timeout_s=2.),dict(late_ack_timeout_s=31)]:
        with pytest.raises(ValueError):validate(cfg)
