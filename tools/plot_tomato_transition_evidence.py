"""Render evidence ranking and the unchanged hard early-transition case."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from . import tomato_dataset as ds
from .train_tomato_transition_evidence import BASE


def run(root,out):
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11});out.mkdir(parents=True,exist_ok=True);r=ds.read_json(root/'results.json');arms=['E0','E1','E2'];colors=['#5D8DC2','#E6A64A','#399A82']
    fig,axes=plt.subplots(1,2,figsize=(13,5),facecolor='#F6F8FC');fig.subplots_adjust(top=.73,bottom=.23,left=.08,right=.97,wspace=.35)
    for k,co in enumerate(['oof10_base_unseen_exposed','development8_exposed']):
        for j,arm in enumerate(arms):
            values=[s[arm]['evaluation'][co]['near_boundary_05']['auroc'] for s in r.values()];x=k+(j-1)*.22
            axes[0].bar(x,np.mean(values),width=.19,color=colors[j],label=arm if k==0 else None);axes[0].scatter(np.repeat(x,3),values,color='#263648',s=15,zorder=3)
    axes[0].set(ylim=(.5,1),ylabel='边界前后0.5秒 AUROC',xticks=[0,1],xticklabels=['B0未见10条（已被研究使用）','开发8条（已反复使用）']);axes[0].legend(ncol=3,frameon=False);axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True)
    veto=ds.read_json(root/'shadow_veto_diagnostic.json')['development8_exposed'];early=[veto[a]['0.8']['early_rejected'] for a in arms];post=[veto[a]['0.8']['post_boundary_rejected'] for a in arms];x=np.arange(3)
    axes[1].bar(x-.16,early,.3,label='会挡住的提前确认（共8次）',color='#399A82');axes[1].bar(x+.16,post,.3,label='也会挡住的已过边界确认（共136次）',color='#D46B61');axes[1].set(xticks=x,xticklabels=arms,ylabel='三种子累计次数');axes[1].legend(fontsize=9,frameon=False);axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
    for container in axes[1].containers:axes[1].bar_label(container,padding=3,fontsize=9)
    fig.text(.06,.93,'观测变化有增量信号，但尚不适合直接控制切换',fontsize=21,weight='bold',color='#203047');fig.text(.06,.84,'E0：概率历史   E1：观测变化   E2：两者组合；右图固定分数门槛0.8，仅为B0提交时刻的旁路诊断。',fontsize=11,color='#526176');fig.text(.06,.08,'AUROC是排序指标，不是正确率。黑点为三个种子；同一录制跨种子重复，不是独立测试。',fontsize=10,color='#526176');fig.text(.06,.035,'没有改动B0、Prompt或原始Subtask；右图不等于动态门控回放，也不代表最终漏切换数量。',fontsize=10,color='#526176');fig.savefig(out/'evidence_summary.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
    eid='3e645bf4e137d0f836c9d7ff';times=np.load(BASE/'seed42'/f'{eid}_input.npz')['times'];boundary=2227/30
    fig,ax=plt.subplots(figsize=(12.5,5),facecolor='#F6F8FC');fig.subplots_adjust(top=.77,bottom=.23,left=.09,right=.97)
    for a,c in zip(arms,colors):
        z=np.load(root/'seed42'/f'{a}_{eid}.npz');ax.plot(times-boundary,z['six_phase_evidence'][:,3],label=a,lw=2,color=c)
    ax.axvline(0,color='#25364A',ls='--',label='人工边界');ax.axvline(72-boundary,color='#D46B61',label='B0提前2.23秒确认');ax.axhline(.8,color='#8993A2',ls=':',label='诊断门槛0.8');ax.set(xlim=(-5,3),ylim=(0,1.03),xlabel='相对人工边界时间（秒）',ylabel='“下一阶段已经开始”的分数');ax.grid(alpha=.15);ax.legend(loc='lower right',ncol=2,fontsize=10);fig.text(.06,.92,'典型adjust→pickup_2提前案例仍未解决',fontsize=21,weight='bold',color='#203047');fig.text(.06,.83,'固定候选阶段的旁路输出 · seed42 · B0确认时E2分数约0.809，仍会通过0.8门槛。',fontsize=11,color='#526176');fig.text(.06,.07,'该图固定观察候选头，不用真实阶段驱动在线状态；分数来自平衡样本训练，尚不是校准后的实际正确概率。',fontsize=10,color='#526176');fig.savefig(out/'adjust_case.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.run,a.out)
