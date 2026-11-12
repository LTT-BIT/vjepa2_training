import numpy as np
from tools.compare_tomato_paired_decisions import colleague_replay,b0_replay,score


def p(k):
    a=np.full(4,.01);a[k]=.97;return a


def test_future_only_commit_and_post_ack_timeout():
    ts=np.arange(0,4,.5);c=np.stack([p(0)]*len(ts));f=np.stack([p(1)]*len(ts))
    a,meta=colleague_replay(c,f,ts,4,1,True);assert a==[dict(time=.5,phase=1)] and meta['timeouts']==1
    assert b0_replay(c,f,ts,4,True)[0]==[]


def test_same_clock_b0_waits_for_two_fresh_observations():
    ts=np.arange(0,2,.5);c=np.stack([p(1)]*len(ts));f=c.copy()
    assert b0_replay(c,f,ts,2,False)[0]==[dict(time=.5,phase=1)]
    assert b0_replay(c,f,ts,2,True)[0]==[dict(time=1.,phase=1)]


def test_seven_terminal_and_no_oracle_reset():
    labels=np.repeat([0,1,2,3,0,1,2,3],4);ts=np.arange(len(labels))*.5;c=np.stack([p(k) for k in labels])
    commits,_=colleague_replay(c,c,ts,16,1,False)
    assert len(commits)==6 and commits[-1]['phase']==6


def test_prefix_causality():
    ts=np.arange(0,7,.5);c=np.stack([p(0 if t<2 else 1) for t in ts]);f=np.stack([p(1)]*len(ts));full,_=colleague_replay(c,f,ts,7,1,True)
    short,_=colleague_replay(c[:6],f[:6],ts[:6],3,1,True)
    assert short==[r for r in full if r['time']<3]
