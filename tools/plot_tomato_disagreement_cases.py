"""Plot fixed diagnostic cases from the no-training disagreement replay."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from . import tomato_dataset as ds


def run(root,out):
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11})
    audit=ds.read_json(root/'event_attribution.json');cases=[('3e645bf4e137d0f836c9d7ff',3,'Current提前；Future对未来时刻的预测正确'),('3e645bf4e137d0f836c9d7ff',4,'Current/Future一起过早：分歧检测看不出来'),('0b12ac0fda9d7cd77dee6e60',4,'Current确认滞后：继续加确认次数会更慢')]
    fig,axes=plt.subplots(3,1,figsize=(13.5,10),facecolor='#F6F8FC');fig.subplots_adjust(left=.09,right=.97,top=.86,bottom=.10,hspace=.68)
    names=['拿起','摘取','放回','调整']
    for ax,(eid,ordinal,title) in zip(axes,cases):
        row=next(r for r in audit['events'] if r['seed']==42 and r['episode_id']==eid and r['ordinal']==ordinal);z=np.load(root/'seed42'/f'{eid}_input.npz');target=ds.LABELS.index(row['transition'].split('->')[1]);b=row['boundary_s'];t=z['times']
        ax.plot(t-b,z['current'][:,target],color='#347DCC',lw=2,label='Current：现在属于下一类的概率')
        ax.plot(t-b,z['future'][:,target],color='#2E9C7E',lw=2,label='Future：1.5秒后属于下一类的概率')
        ax.axvline(-1.5,color='#2E9C7E',ls=':',lw=1.5,label='未来目标开始跨过边界（−1.5秒）')
        ax.axvline(0,color='#28364A',ls='--',label='人工阶段边界（0秒）')
        ax.axhline(.8,color='#7C8798',ls=':',alpha=.5)
        offset=row['baseline']['offset_s']
        if offset is not None:ax.axvline(offset,color='#D05252',ls='-',alpha=.8,label=f'B0确认（{offset:+.2f}秒）')
        ax.set(xlim=(-4.5,5.5),ylim=(-.03,1.07),ylabel=f'下一类：{names[target]}\n概率',xlabel='相对人工边界时间（秒；负数为提前）')
        ax.set_title(title,loc='left',fontsize=13,pad=10);ax.grid(alpha=.12);ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',fontsize=8,framealpha=.9,ncol=2)
    fig.text(.06,.953,'为什么不能只看 Current 与 Future 是否一致来切换？',fontsize=22,weight='bold',color='#203047')
    fig.text(.06,.917,'真实开发录制 · 冻结B0 seed42 · 所有曲线按观测时刻绘制 · 未修改人工标注',fontsize=12,color='#526176')
    fig.text(.06,.034,'这里是离线模型证据与阶段确认；没有ROSbag或机器人执行时序，红线不代表机器人已经执行新动作。',fontsize=10,color='#526176')
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=155,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.run,a.out)
