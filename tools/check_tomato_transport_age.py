"""Reproduce existing bridge/Shadow age-limit mismatch without changing either."""
import argparse
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds
from .tomato_frozen_protocol import verify
from .check_tomato_frozen_openpi import load_snapshot
from .tomato_offline_contract import prediction_packet
from .tomato_task_memory import TaskMemory,Observation


def run(lock_path,out):
    if out.exists():raise FileExistsError('New diagnostic directory required')
    lock=verify(lock_path);out.mkdir(parents=True);bridge=load_snapshot(lock,out/'snapshot');results=[]
    for age in [.05,1.5,2.5]:
        session=f'age-{age}';model=lock['models']['42']['model_version'];provider=bridge.PredictionPromptProvider();provider.reset(session);memory=TaskMemory(session,model,mode='M1');rows=[]
        c=np.array([.001,.997,.001,.001]);f=c.copy()
        for i in range(2):
            t=i*.5;oid=f'{session}:{i}';packet=prediction_packet(bridge,c,f,session,oid,t,model)
            accepted=True
            try:context=provider.accept_prediction(packet,observation_id=oid,now=t+age)
            except ValueError:accepted=False
            view=memory.observe(Observation(session,i,oid,t,t+age,tuple(c),tuple(f),model),b0_phase=int(provider.phase))
            rows.append(dict(observation_timestamp=t,age_s=age,bridge_accepted=accepted,bridge_phase=provider.phase,memory_valid=view['valid'],memory_phase=view['confirmed_phase']))
        results.append(dict(age_s=age,rows=rows,phase_mismatch=provider.phase!=memory.phase))
    assert not results[0]['phase_mismatch'] and results[1]['phase_mismatch'] and not results[2]['phase_mismatch']
    ds.write_json(out/'result.json',dict(cases=results,bridge_max_age_s=2.,memory_max_age_s=1.,defaults_changed=False,
                  issue='Between1s and2s, pinned bridge can confirm Current while Shadow rejects the observations; fresh replay fidelity is not a universal transport guarantee.',
                  next_step='Before live integration explicitly unify ingress-age policy and replay delayed messages. This diagnostic does not tighten frozen B0 thresholds.',
                  lock_sha256=ds.digest(lock_path),source_sha256=ds.digest(__file__)))
    print('REPRODUCED: clean .05s aligned;1.5s bridge advances but Shadow rejects;2.5s both reject. No defaults changed.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,default=Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.lock,a.out)
