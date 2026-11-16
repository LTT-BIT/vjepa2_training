"""Inspect native versus actual last-frame local input; labels are plot-only."""
import argparse
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds


def run(cache):
    import av,cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    manifest=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json');progress=ds.read_json(cache/'progress.json');entry=next(r for r in progress['episodes'] if r['split']=='train');e=next(e for e in manifest['episodes'] if e['episode_id']==entry['episode_id']);z=np.load(entry['path']);assert ds.digest(entry['path'])==entry['sha256'];request=[]
    for seg in e['annotation']['segments']:
        midpoint=(seg['start_frame']+seg['end_frame'])//2;k=int(np.argmin(abs(z['anchors']-midpoint)));request.append((k,seg['label']))
    wanted={int(z['anchors'][k]) for k,_ in request};frames={}
    with av.open(e['videos'][0]) as video:
        for i,frame in enumerate(video.decode(video=0)):
            if i in wanted:frames[i]=frame.to_ndarray(format='rgb24')
    cfg=ds.read_json(cache/'protocol.json');recorded={}
    if cfg.get('roi_source_mode')=='recorded_offline_r11':
        root=Path(cfg['roi_bundle']);row=next(r for r in ds.read_json(root/'bundle.json')['mapping'] if (r['source_dataset'],r['source_episode_index'])==(e['dataset_root'],e['episode_index']))
        path=root/f'b_coarse/videos/chunk-000/observation.images.interaction/episode_{row["episode_index"]:06d}.mp4'
        with av.open(path) as stream:
            for i,frame in enumerate(stream.decode(video=0)):
                if i in wanted:recorded[i]=cv2.resize(frame.to_ndarray(format='rgb24'),(256,256),interpolation=cv2.INTER_AREA)
    fig,axes=plt.subplots(2,len(request),figsize=(17,6))
    for col,(k,label) in enumerate(request):
        anchor=int(z['anchors'][k]);im=frames[anchor];h,w=im.shape[:2];box=np.rint(z['metadata'][k,60:64]*np.array([w,h,w,h])).astype(int);l,t,r,b=box;native=np.zeros((r-l,r-l,3),np.uint8);x1,y1,x2,y2=max(l,0),max(t,0),min(r,w),min(b,h);native[y1-t:y2-t,x1-l:x2-l]=im[y1:y2,x1:x2];crop=cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA);valid=bool(z['roi_valid'][k,-1]);axes[0,col].imshow(im);axes[0,col].add_patch(Rectangle((l,t),r-l,b-t,fill=False,color='lime' if valid else 'orange'));axes[0,col].set_title(f'{label} / {anchor/30:.1f}s',fontsize=9);axes[1,col].imshow(recorded.get(anchor,crop));axes[1,col].set_title('Recorded r11 ROI' if recorded else ('ROI candidate' if valid else 'Fresh full-view fallback'),fontsize=9)
        for row in range(2):axes[row,col].axis('off')
    fig.suptitle('Old100 TRAIN example: native cam0 (top) / actual local256 last frame (bottom)\nLabels and overlay are diagnostic only; NOT model input. Encoder uses16 past/current pixel timestamps; recorded r11 boxes are future-assisted offline.',fontsize=11);fig.tight_layout();path=cache/'local_input_example.png';fig.savefig(path,dpi=130);plt.close(fig);print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path);run(p.parse_args().cache)
