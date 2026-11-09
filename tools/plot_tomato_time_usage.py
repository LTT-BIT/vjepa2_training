from pathlib import Path
import io,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from tools import tomato_dataset as ds
root=Path('runs/tomato/time_usage_analysis_20261109');x=ds.read_json(root/'analysis.json');s=x['seeds']['42'];fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for key,label in [('time_risk','Time risk ranking'),('future_confidence','Future confidence ranking')]:
 c=s['ranking'][key]['curve'];ax[0].plot([v['recall'] for v in c],[v['precision'] for v in c],label=label)
ax[0].set(xlabel='Recall among B0 matched events',ylabel='Candidate precision',title='Ranking only; online warning set stays unchanged');ax[0].legend(fontsize=8)
th=list(s['threshold_sweep_not_selection']);ax[1].plot([s['threshold_sweep_not_selection'][t]['false_per_minute'] for t in th],[s['threshold_sweep_not_selection'][t]['recall'] for t in th],marker='o')
for t in th:ax[1].annotate(t,(s['threshold_sweep_not_selection'][t]['false_per_minute'],s['threshold_sweep_not_selection'][t]['recall']))
ax[1].set(xlabel='False warnings/min',ylabel='Strict event recall',title='Hard gating sweep: diagnostic, not threshold selection');fig.savefig(root/'ranking_and_tradeoff.png',dpi=160);plt.close(fig)
cases=ds.read_json('runs/tomato/boundary_review_20261109/review_package/cases.json')['cases']
for event in ds.read_json(root/'adjust_cases.json'):
 folder=Path(event['traces']).parent;f=np.load(event['traces']);b=event['boundary_s'];t=f['prediction_times']-b;fig,axs=plt.subplots(3,1,figsize=(11,8),constrained_layout=True)
 axs[0].plot(t,f['current'][:,0],label='Current pickup');axs[0].plot(t,f['future'][:,0],label='Future pickup');axs[0].plot(t,1-f['time_distribution'][:,-1],label='P(transition <=1.5s)');axs[0].axhline(.5,color='gray',ls=':');axs[0].legend(fontsize=8);axs[0].set_ylim(0,1.03)
 st=f['state_times']-b;axs[1].plot(st,np.sqrt(np.mean(f['right_arm_velocities']**2,axis=1)),label='Named right arm7D velocity RMS');axs[1].plot(st,np.sqrt(np.mean(f['right_hand_velocities']**2,axis=1)),label='Named right hand22D velocity RMS');axs[1].legend(fontsize=8)
 for cam in range(3):axs[2].plot(t[1:],np.linalg.norm(np.diff(f['pooled_camera_features'][:,cam],axis=0),axis=1),label='Camera'+str(cam)+' pooled-feature change')
 axs[2].legend(fontsize=8);axs[2].set_xlabel('Seconds relative to provisional human boundary')
 for a in axs:a.axvline(0,color='black',ls='--');a.axvspan(-1.5,0,color='green',alpha=.06)
 fig.suptitle(event['episode_id']+' adjust -> pickup');fig.savefig(folder/'curves.png',dpi=130);plt.close(fig)
 case=next(c for c in cases if c['episode_id']==event['episode_id'] and c['kind']=='truth' and c['from_stage']=='adjust' and c['to_stage']=='pickup');video=Path('runs/tomato/boundary_review_20261109/review_package')/case['video'];frames=[]
 for dt in [-.5,0,.5]:
  sec=b+dt;data=subprocess.check_output(['ffmpeg','-v','error','-ss',str(sec-case['clip_start_s']),'-i',str(video),'-frames:v','1','-f','image2pipe','-vcodec','png','-threads','1','-']);im=Image.open(io.BytesIO(data)).convert('RGB');im.thumbnail((1000,400));frames.append((sec,im.copy()))
 sheet=Image.new('RGB',(1000,sum(im.height+25 for _,im in frames)),'white');draw=ImageDraw.Draw(sheet);y=0
 for sec,im in frames:draw.text((5,y+5),f'Source time {sec:.3f}s; human proposal {b:.3f}s',fill='black');sheet.paste(im,(0,y+25));y+=im.height+25
 sheet.save(folder/'keyframes.png')
print('Eight full difficult-case curve/frame packs generated')
