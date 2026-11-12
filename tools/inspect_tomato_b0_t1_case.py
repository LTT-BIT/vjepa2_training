"""Read-only video/state evidence for a persistent early transition; no relabeling."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
from . import tomato_dataset as ds,tomato_phase_model as pm


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json');e=next(e for e in manifest['episodes'] if e['episode_id']=='3e645bf4e137d0f836c9d7ff');boundary=e['annotation']['segments'][4]['start_frame']/e['fps'];out=root/'persistent_adjust_pickup';out.mkdir(exist_ok=False)
    requested=np.array([-4,-3,-2,-1,0,1,2]);frames=np.rint((boundary+requested)*e['fps']).astype(int);views,_=pm.decode_selected(e,frames);fig,axes=plt.subplots(3,len(frames),figsize=(16,7))
    for camera in range(3):
        for k,frame in enumerate(frames):
            ax=axes[camera,k];ax.imshow(views[camera][int(frame)]);ax.set_xticks([]);ax.set_yticks([])
            if camera==0:ax.set_title(f'{frame/e["fps"]:.3f}s ({frame/e["fps"]-boundary:+.1f}s)',fontsize=9)
            if k==0:ax.set_ylabel(e['camera_keys'][camera],fontsize=8)
    fig.suptitle('A/1103_620_642_normal_lerobot/episode_000000 | human adjust->pickup boundary 74.233s\nActual source frames; offsets relative to unchanged annotation. No success/failure inferred.',fontsize=11);fig.tight_layout();fig.savefig(out/'source_frames.png',dpi=140);plt.close(fig)
    times,q=ds.read_state(e,manifest['contract']['joint_names'],manifest['contract']['joint_mode']);velocity=np.zeros_like(q);velocity[1:]=np.diff(q,axis=0)/np.diff(times)[:,None];names=manifest['contract']['joint_names'];arm=[i for i,n in enumerate(names) if n.startswith('right_fr3_')];hand=[i for i,n in enumerate(names) if n.startswith('right_') and not n.startswith('right_fr3_')];assert len(arm)==7 and len(hand)==22
    mask=(times>=boundary-6)&(times<=boundary+6);fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    for col in arm:axes[0].plot(times[mask]-boundary,q[mask,col],label=names[col].split('.')[0],lw=.8)
    axes[0].set_ylabel('Right arm rad');axes[0].legend(ncol=4,fontsize=7)
    axes[1].plot(times[mask]-boundary,np.linalg.norm(velocity[mask][:,arm],axis=1),label='Right arm velocity norm');axes[1].plot(times[mask]-boundary,np.linalg.norm(velocity[mask][:,hand],axis=1),label='Right hand velocity norm');axes[1].set_ylabel('rad/s norm');axes[1].legend(fontsize=8)
    for col in hand:axes[2].plot(times[mask]-boundary,q[mask,col],lw=.6)
    axes[2].set_ylabel('Right 22 hand rad')
    z=np.load(root/'seed42'/f'{e["episode_id"]}.npz');m=(z['times']>=boundary-6)&(z['times']<=boundary+6)
    for key,color in [('b0_current','#2563eb'),('t1_current','#dc2626'),('future','gray')]:axes[3].plot(z['times'][m]-boundary,z[key][m,0],marker='.',label=key+' pickup',color=color)
    axes[3].set_ylabel('Pickup probability');axes[3].legend(fontsize=8)
    for ax in axes:ax.axvline(0,color='black',ls='--');ax.grid(alpha=.2)
    axes[-1].set_xlabel('Seconds relative to unchanged human boundary');fig.tight_layout();fig.savefig(out/'state_and_probabilities.png',dpi=140);plt.close(fig)
    np.savez_compressed(out/'measured_state.npz',times=times[mask],positions=q[mask],backward_velocity=velocity[mask],joint_names=np.array(names))
    start=boundary-6;cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n']
    for video in e['videos']:cmd+=['-ss',str(start),'-t','12','-i',video]
    filters=[f'[{i}:v]fps=15,scale=480:360:force_original_aspect_ratio=decrease,pad=480:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]' for i in range(3)];filters.append('[v0][v1][v2]hstack=inputs=3[out]')
    cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','23',str(out/'three_camera_clip.mp4')];subprocess.run(cmd,check=True)
    ranges={}
    for a,b in [(-4,-2),(-2,0),(0,2)]:
        m=(times>=boundary+a)&(times<boundary+b);ranges[f'{a}:{b}']=dict(right_arm_mean_speed_norm=float(np.linalg.norm(velocity[m][:,arm],axis=1).mean()),right_hand_mean_speed_norm=float(np.linalg.norm(velocity[m][:,hand],axis=1).mean()),samples=int(m.sum()))
    case=dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],dataset_root=e['dataset_root'],episode_index=e['episode_index'],human_boundary_s=boundary,annotation_sha256=e['annotation']['sha256'],clip_source_start_s=start,clip_boundary_offset_s=6.,camera_order=e['camera_keys'],source_frame_indices=frames.tolist(),state_bins=ranges,interpretation='Motion statistics and source imagery are descriptive, not independent operation-result labels. No automatic boundary repair.',source_sha256={p:ds.digest(p) for p in e['videos']+[e['parquet']]})
    ds.write_json(out/'case.json',case);print('CASE',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);run(p.parse_args().run)
