"""All development seed42 boundaries, without selecting only improved examples."""
import argparse
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=ds.read_json(root/'protocol.json');data=ds.read_json(root/'evaluations_with_b0.json')['42'];manifest=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json');methods=['B0_frozen']+cfg['methods'];classes=[0,1,2,3,0,1,2];folder=root/'boundary_curves_seed42';folder.mkdir(exist_ok=True);catalog=[]
    for e in manifest['episodes']:
        if e['split']!='val':continue
        eid=e['episode_id'];z=np.load(Path('runs/tomato/disagreement_20261112_v2/seed42')/f'{eid}_input.npz');times=z['times'];pred={'B0_frozen':z['current']}
        for method in cfg['methods']:pred[method]=np.load(root/'seed42'/f'{method}_{eid}.npz')['current']
        for ordinal,segment in enumerate(e['annotation']['segments'][1:],1):
            boundary=segment['start_frame']/e['fps'];target=classes[ordinal];before=classes[ordinal-1];events={m:next(x for x in data[m]['evaluation']['development8_exposed']['events'] if x['episode_id']==eid and x['ordinal']==ordinal) for m in methods}
            offsets=[x['offset_s'] for x in events.values() if x['offset_s'] is not None];left=boundary+min([-4]+offsets)-.5;right=boundary+max([4]+offsets)+.5;select=(times>=left)&(times<=right)
            fig,axes=plt.subplots(4,1,figsize=(10,7),sharex=True)
            for ax,method in zip(axes,methods):
                ax.plot(times[select]-boundary,pred[method][select,target],label='Current next',color='#2563eb',marker='.',markersize=3)
                ax.plot(times[select]-boundary,pred[method][select,before],label='Current previous',color='#d97706')
                ax.plot(times[select]-boundary,z['future'][select,target],label='Frozen Future next (1.5s)',color='gray',alpha=.65)
                ax.axvline(0,color='black',ls='--',lw=1,label='Human boundary');ax.axhline(.8,color='black',ls=':',lw=.6)
                event=events[method]
                if event['offset_s'] is not None:ax.axvline(event['offset_s'],color='#dc2626',lw=1,label='Prompt confirmation')
                ax.set_ylim(-.02,1.02);ax.set_ylabel(method);ax.set_title(event['outcome']+'; offset='+('missing' if event['offset_s'] is None else f"{event['offset_s']:+.3f}s"),fontsize=8,loc='right');ax.grid(alpha=.15)
            axes[0].legend(ncol=3,fontsize=7);axes[-1].set_xlabel('Seconds relative to existing human boundary (diagnostic reference only)');fig.suptitle(f"{e['source_recording_id']} | event {ordinal}: {events['B0_frozen']['transition']}",fontsize=9);fig.tight_layout();path=folder/f'{eid}_event{ordinal}.png';fig.savefig(path,dpi=130);plt.close(fig)
            catalog.append(dict(episode_id=eid,source_recording_id=e['source_recording_id'],ordinal=ordinal,boundary_s=boundary,events=events,figure=str(path)))
    ds.write_json(root/'boundary_curve_catalogue.json',catalog);print('PLOTS',len(catalog))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
