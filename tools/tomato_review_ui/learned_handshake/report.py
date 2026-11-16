"""Locked val selection, source-paired uncertainty, raw-event audit and handoff."""
from pathlib import Path
import json,zipfile,time
import numpy as np
import torch
from .contracts import *
from .evaluate import csv_rows,bootstrap
from .models import load_checkpoint,LearnedPolicy
from .targets import label


def diagnostics(c,out,names):
 reports={};hazards={};attribution=[]
 metadata={e['episode_id']:e for e in read(Path(c['baseline_root'])/'joint/manifest.json')['episodes']}
 for name in names:
  pp=[];yy=[];action_cm=np.zeros((5,5),int);time_cm=np.zeros((6,6),int);brier=[];time_probs=[];time_y=[];errors=[];buckets={};held_near=0;near=0
  for path in (out/'predictions'/name/'val').glob('*.jsonl'):
   meta=metadata[path.stem]
   for line in path.open():
    r=json.loads(line);f=r['features'];f['sequence']=np.asarray(f['sequence']);target=label(meta,f)
    if not target['supervision_valid']:continue
    proposal=r['proposal'];diag=r['diagnostics'];actual=(LABELS.index(proposal['candidate_stage'])+1) if proposal['request_commit'] and proposal['candidate_stage'] is not None else 0
    action_cm[target['action'],actual]+=1
    if target['delta_s'] is not None:
     d=-target['delta_s'];key='stable'
     for lo,hi in [(-2,-1),(-1,-.5),(-.5,0),(0,.5),(.5,1),(1,2)]:
      if lo<=d<hi:key=f'[{lo},{hi})';break
     v=buckets.setdefault(key,dict(samples=0,requests=0,positive=0));v['samples']+=1;v['requests']+=proposal['request_commit'];v['positive']+=target['action']>0
    if 'gate_probability' in diag:pp.append(diag['gate_probability']);yy.append(target['gate'])
    elif 'action_probabilities' in diag:pp.append(1-diag['action_probabilities'][0]);yy.append(target['action']>0)
    if 'time_probabilities' in diag and target['time_bin']>=0:
     tp=np.asarray(diag['time_probabilities']);true=target['time_bin'];time_cm[true,tp.argmax()]+=1;time_probs.append(tp);time_y.append(true);brier.append(float(((tp-np.eye(6)[true])**2).sum()))
     if target['delta_s'] is not None and abs(target['delta_s'])<=1:
      predicted=float(tp@np.array([1.5,.75,.25,-.1,-.6,0]));errors.append(abs(predicted-target['delta_s']));near+=1;held_near+=not proposal['request_commit']
  prauc=None;binary_brier=None;binary_ece=None
  if pp:
   pp=np.array(pp);yy=np.array(yy);ii=np.argsort(-pp);tp=np.cumsum(yy[ii]);precision=tp/np.arange(1,len(tp)+1);prauc=float((precision*yy[ii]).sum()/max(1,yy.sum()));binary_brier=float(((pp-yy)**2).mean());binary_ece=0
   for lo in np.arange(0,1,.1):
    keep=(pp>=lo)&(pp<lo+.1 if lo<.9-1e-9 else pp<=1)
    if keep.any():binary_ece+=keep.mean()*abs(pp[keep].mean()-yy[keep].mean())
  reports[name]=dict(action_confusion=action_cm.tolist(),request_PR_AUC_average_precision=prauc,request_brier=binary_brier,request_ece=binary_ece,distance_buckets=buckets)
  if time_y:
   tp=np.array(time_probs);ty=np.array(time_y);conf=tp.max(1);ok=tp.argmax(1)==ty;ece=0
   for lo in np.arange(0,1,.1):
    keep=(conf>=lo)&(conf<lo+.1 if lo<.9-1e-9 else conf<=1)
    if keep.any():ece+=keep.mean()*abs(conf[keep].mean()-ok[keep].mean())
   hazards[name]=dict(time_confusion=time_cm.tolist(),supports=time_cm.sum(1).tolist(),brier=float(np.mean(brier)),ece=float(ece),near_boundary_MAE_s=float(np.mean(errors)) if errors else None,near_boundary_hold_fraction=held_near/max(near,1),near_boundary_samples=near)
  for path in (out/'replays'/name/'val').glob('*.json'):
   r=read(path)['replay'];events=r['events']
   for ev in events:
    if ev['kind'] in ['policy_proposal','request','commit','ack','late_ack','hard_timeout']:attribution.append(dict(strategy=name,episode_id=path.stem,**ev))
 write(out/'summary/model_calibration.json',reports);write(out/'summary/hazard_time_bins.json',hazards)
 with (out/'summary/policy_proposals_and_commit_attribution.jsonl').open('w') as f:
  for r in attribution:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 return reports,hazards


def benchmark(c,out):
 index=read(out/'model_index.json');results={};torch.set_num_threads(1)
 for name,rec in index.items():
  saved,model=load_checkpoint(rec['path'],rec['bindings'],c);x=torch.zeros(1,92);seq=torch.zeros(1,4,11);state=torch.zeros(1,48);mask=torch.ones(1,4,dtype=torch.bool);timings=[]
  with torch.inference_mode():
   for _ in range(20):model(x,seq,state,mask)
   for _ in range(200):
    start=time.perf_counter();model(x,seq,state,mask);timings.append((time.perf_counter()-start)*1000)
  results[name]=dict(cpu_forward_ms={f'p{p}':float(np.percentile(timings,p)) for p in [50,95,99]},parameters=rec['manifest']['parameters'],context_capacity_bytes=4*11*4+48*4,condition='CPU batch1 normalization/context assembly excluded; 1 policy selected, not ensemble',encoder_pi05_concurrency='NOT_MEASURED')
 write(out/'summary/latency.json',results);return results


def generate(c,out):
 summary=read(out/'summary/val_overall.json');selection=read(out/('selection.corrected.lock.json' if (out/'selection.corrected.lock.json').exists() else 'selection.lock.json'));historical=read(out/'summary/historical_test_locked.json');base=summary['H0'];rows=[];source_rows=[];transitions=[];paired={}
 for name,r in summary.items():
  rows.append(dict(strategy=name,strict_hits=r['strict_hits'],eligible=r['eligible'],wide_hits=r['matched'],missed=r['missed'],unmatched=r['unmatched_commits'],early_ge05=r['early_ge05'],late_gt05=r['late_gt05'],wrong_seconds=r['wrong_active_s'],mean_wrong_fraction=r['mean_wrong_fraction'],exact_order=r['exact_order'],hard_timeouts=r['hard_timeouts'],recovery_seconds=r['recovery_required_s'],policy_requests=r['policy_requests'],mask_rejections=r['task_mask_rejections']))
  paired[name]=bootstrap(r['by_source'],base['by_source'],c['bootstrap_replicates'],c['bootstrap_seed'])
  for e in r['by_source']:
   source_rows.append(dict(strategy=name,episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],strict_hits=e['strict_hits'],wrong_seconds=e['wrong_active_s'],early_ge05=e['early_ge05'],missed=e['missed'],unmatched=e['unmatched_commits'],hard_timeouts=e['hard_timeouts']))
   raw=read(out/'replays'/name/'val'/(e['episode_id']+'.json'))['replay']['events']
   for event in e['transitions']:
    commit=next((v for v in raw if v['kind']=='commit' and event['commit_time'] is not None and abs(v['time']-event['commit_time'])<1e-8),None)
    related=[v for v in raw if commit and v['prompt_version']==commit['prompt_version'] and v['time']>=commit['time']]
    ack=next((v for v in related if v['kind'] in ['ack','late_ack']),None)
    transitions.append(dict(strategy=name,episode_id=e['episode_id'],**event,ack_kind=ack['kind'] if ack else None,effective_ack_time=ack['time'] if ack else None,soft_timeout_at=next((v['time'] for v in related if v['kind']=='timeout'),None),hard_timeout_at=next((v['time'] for v in related if v['kind']=='hard_timeout'),None)))
 write(out/'summary/paired_bootstrap.json',paired);csv_rows(out/'summary/val_by_seed.csv',rows);csv_rows(out/'summary/val_by_source.csv',source_rows);csv_rows(out/'summary/val_by_transition.csv',transitions);csv_rows(out/'summary/early_late_hard_timeout.csv',rows)
 names=[n for n in summary if n not in ['H0','R1','R2']];cal,haz=diagnostics(c,out,names);lat=benchmark(c,out)
 lines=['# 双头握手决策受控实验报告（V2，2026-11-12）','',f'实验：`{out}`。基线 `db4cdad6e87af7c077f5be16a00bc277efeb54d9`；分支 `feat/tomato-learned-handshake-v2-20261112`。未提交/推送，原 GUI/分类器/握手未修改。','',f'**验证集锁定结论：{selection["selected"]}。** 选型先看非法实际commit、硬超时、过早、漏切及额外提交，再看严格命中、Wrong Active与完整顺序。不能以Wrong Active下降宣称更安全。','', '## 实验身份与比较含义','', '93条原normal来源，75/9/9固定划分。仅原B0 Joint seed42双头概率；三种子只改变策略网络训练，绝不混用上游三种子。三路完整RGB/58D/+1s/2Hz契约不变，本轮不解码/编码视频，也未训练任何视觉、Memory或Joint teacher。五折OOF复用上一轮已验证54 train/6 inner-val/15 held教师，val/test为固定全75训练B0。','', '主概率使用上一轮B0-frozen保存值，来自同一个原B0 checkpoint；其先前与原分split分批概率最大浮点误差5.64e-7。本轮H0与这些历史事件、时间戳、Active逐值相等；val必须复现23/54窄窗与98.233秒。任何性能结果均由本轮实际回放产生。','', 'H0原码直接调用；R1/R2及学习器复用原Handshake的ACK、timer和planning，通过独立REQUEST注入点替换候选逻辑。ACK输入的Future暂时中和为Active以抑制旧候选旁路，随后恢复真实Evidence；不改变原概率源或Current ACK判断。任务mask是外部四类图，单测覆盖非NEXT图和空图。Active只能原planner原子提交，阻塞/TTL/身份约束全部保留。','', '## 验证集主表（9来源、54边界）','', '|策略|窄窗|宽窗|漏切|未匹配commit|提前≤-.5s|晚于.5s|Wrong Active(s)|顺序正确|硬超时|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,r in summary.items():lines.append(f"|{name}|{r['strict_hits']}/54|{r['matched']}|{r['missed']}|{r['unmatched_commits']}|{r['early_ge05']}|{r['late_gt05']}|{r['wrong_active_s']:.3f}|{r['exact_order']}/9|{r['hard_timeouts']}|")
 lines+=['','提前/迟到仅按宽窗[-1,+2]一对一真实from/to匹配事件计数，窄窗是其中[-.2,+.5]的子集；未匹配提交单列。Wrong Active以完整Episode与GT四粗类交叠积分，逐Episode等权比例另见CSV。三个seed不是独立来源。','', '## 模型、训练与状态分布','', 'L0为56维显式L2 Logistic（BCE+.5*.01*||W||²、不惩罚bias，优化器decay0）；L1a为候选MLP门控；L1b为92→64→32→5动作；L2为masked GRU32加相同状态后小读出；L3a与L1b同MLP trunk，next4/time6输出。L0/L1a仅候选受Future≥.65限定；L1b/L2/L3a可主动提出目标，不在旧H0已请求之后才veto。','', '模型AdamW 3e-4（L0用显式L2、optimizer decay0；其余WD.01）、batch256、最多80epoch、patience10、dropout.1、clip1；train-only归一化和事件/来源/时间桶加权。checkpoint仅按固定val-H0状态监督损失早停，之后才进行自身Active完整回放。自回放训练状态若相对H0 Active时间偏移>.10，只做一次预锁定train-only轨迹聚合，再同seed重训；触发则直接使用v1，不按val Active挑v0/v1。','', '## 核心问题逐项回答','']
 def means(group,key):return float(np.mean([summary[f'{group}_s{s}'][key] for s in c['seeds']]))
 lines.append(f"1. 固定B0输入，最终按风险条件选择 {selection['selected']}；完整准入审计在selection.lock.json。所有分类Current/Future F1保持完全一致，变化只来自决策。")
 lines.append(f"2. 简单规则：R1窄窗{summary['R1']['strict_hits']}、Wrong {summary['R1']['wrong_active_s']:.3f}s；R2窄窗{summary['R2']['strict_hits']}、Wrong {summary['R2']['wrong_active_s']:.3f}s。相对H0须连同表中过早/硬超时判断，不能因神经网络更复杂而优先选择。")
 lines.append(f"3. L1a只学时机：三seed窄窗均值{means('L1a','strict_hits'):.2f}，Wrong {means('L1a','wrong_active_s'):.3f}s；L1b同时选目标：{means('L1b','strict_hits'):.2f}、{means('L1b','wrong_active_s'):.3f}s。非法目标提议被外部mask拒绝，真实非法commit须为0，拒绝率和动作混淆另列。")
 lines.append(f"4. L2 GRU窄窗均值{means('L2','strict_hits'):.2f}、Wrong {means('L2','wrong_active_s'):.3f}s，对照L1b；只读4个2Hz概率锚点（跨度1.5s），没有读取原视觉/关节Memory，也没有验证更长上下文。")
 lines.append(f"5. L3a时间hazard窄窗均值{means('L3a','strict_hits'):.2f}、Wrong {means('L3a','wrong_active_s'):.3f}s。同backbone L1b对照；time概率好看不代表commit更准。*_mass仅为锁定CDE质量映射消融，不参与主选型。")
 lines+=['6. 归因按policy_proposal→accepted request→commit→ACK保存：提前误切需同时看原Future、提议时刻与planner边界；漏切需区分HOLD、task_mask、ACK/RECOVERY阻断。summary/policy_proposals_and_commit_attribution.jsonl与casebook保留原始证据；不凭GT复位。','7. K1仅在本轮主策略锁定后，以现有seed42 Future做固定输入替换；结果或NOT_RUN理由见summary/k1_future_swap.json，不用它再选策略。','8. 参数与单次CPU P50/P95/P99见latency.json；只加载一个被选择策略，不是集成15个网络。Encoder+π0.5并发、实机物理完成均NOT_MEASURED/NOT_EVALUABLE。离线回放不是机器人闭环测试；若风险未通过，不建议改变默认进入控制执行。','', '## 监督、hazard与边界不确定性','', '监督在实际下一planner提交tick计算，严格窗commit−GT∈[-.2,+.5]；已迟到但GT当前仍为合法候选时保留late_catchup正例。hazard A/B/C/D/E/F严格按协议分桶；迟到超过1秒但仍可追赶的样本time loss被mask，行动监督仍为正，绝不冒充F负例。末尾下一个边界右删失时屏蔽next/time；commit tick仍有当前GT则保留HOLD行动监督；录制范围外没有监督，不填F负例，更不代表物理FINISH。','', 'L3固定cost：switch=[4,3,2,0,.3,4]，hold=[0,.25,1,3,4,0]，next≥.5且期望switch<hold才请求。没有依据val/test调整cost。time混淆、Brier/ECE、near-boundary MAE在hazard_time_bins.json；纯监督分数不能代替Active评价。','', '只对train标签做±1帧和±.1秒扰动审计，变化数量见train_labels_audit.json；原标注未修改。OOF与全75训练概率质量、置信度直方图、分歧、边界附近质量见teacher_quality.json；teacher质量/分布漂移可能限制learned策略，不把负结果解释为神经策略普遍无效。','', '## 历史保留集诊断','', 'val选型和权重SHA锁定后，仅H0与唯一候选评估。旧test已多次曝光，不是新盲测，未用于更改架构、阈值、代价或默认策略。','', '|策略|窄窗|漏切|未匹配|提前≤-.5s|Wrong(s)|硬超时|','|---|---:|---:|---:|---:|---:|---:|']
 for name,r in historical.items():lines.append(f"|{name}|{r['strict_hits']}/{r['eligible']}|{r['missed']}|{r['unmatched_commits']}|{r['early_ge05']}|{r['wrong_active_s']:.3f}|{r['hard_timeouts']}|")
 lines+=['', '## 跨来源不确定性、最坏情况和局限','', '5000次来源聚类配对bootstrap固定seed；val只有9来源且被多轮观察，区间只是描述，不构成显著泛化/安全认证。所有seed和失败方案都保留，最差seed/来源不删除。各模型HOLD比例、正例召回、Action混淆/PR-AUC和hazard bins详见JSON；直接行为克隆若塌缩为HOLD也如实记入。','', 'R3：NOT_RUN，旧V1的预注册累积阈值文件缺失，没有临时自造阈值。L3b和长度8：NOT_RUN_BUDGET，初批五组已覆盖受控最小矩阵，不扩大架构搜索。任务末尾return→adjust仍无物理终止守卫，本轮不能宣称FINISH成功。','', '## 文件、复现与验收','', '新增代码仅在tools/tomato_review_ui/learned_handshake/，新增配置configs/tomato_learned_handshake_v2_20261112.yaml、专项测试和本报告。此前Memory成果保留，没有修改原Current-only/Future-only/Joint训练、GUI、Handshake、原标签或数据。','', '```bash', 'PY=.venv-tomato/bin/python', 'EXP=runs/tomato/learned_handshake_v2_NEW_UNIQUE', '$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --check-only', '$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --phase-a', '$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --smoke', '$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --run', '$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --verify', '$PY -m tools.tomato_review_ui.learned_handshake.cli --output-dir "$EXP" --report-only', '$PY -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py', '```', '', '输出拒绝覆盖既有训练或回放。完整模型保留decision_models/*/seed*/v*/best_policy.pt，SHA见对应manifest/checkpoint.sha256；所有概率、状态和决策事件保留replays/与predictions/，没有打包视频、Encoder、全部特征或大权重。']
 report='\n'.join(lines)+'\n'
 if (out/'ANALYSIS_SUPPLEMENT.md').exists():
  supplement=(out/'ANALYSIS_SUPPLEMENT.md').read_text();lead,tail=supplement.split('## 为什么直接动作网络可能更差',1)
  report=report.replace('## 实验身份与比较含义',lead+'\n## 实验身份与比较含义',1)+'\n## 为什么直接动作网络可能更差'+tail
 path=REPO/'docs/TOMATO_LEARNED_HANDSHAKE_POLICY_ABLATION_REPORT_20261112.md';path.write_text(report);(out/'REPORT.md').write_text(report)
 plot(c,out,summary)
 hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['artifacts_sha256.json','handoff.zip']};write(out/'artifacts_sha256.json',hashes)
 with zipfile.ZipFile(out/'handoff.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if not p.is_file() or p.name=='handoff.zip' or p.suffix=='.pt' or 'predictions' in p.parts or 'smoke' in p.parts:continue
   if 'replays' in p.parts and 'train_oof' in p.parts:continue
   if 'archived_L0_AdamW' in p.parts and any(part.startswith(('predictions_','replays_')) for part in p.parts):continue
   z.write(p,str(Path('evidence')/p.relative_to(out)))
  for p in list(Path(__file__).parent.glob('*.py'))+[REPO/'configs/tomato_learned_handshake_v2_20261112.yaml',REPO/'tests/test_tomato_learned_handshake_v2.py']:z.write(p,str(Path('source')/p.relative_to(REPO)))
 print(path)


def plot(c,out,summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 (out/'plots').mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(13,5))
 groups=['H0','R1','R2','L0','L1a','L1b','L2','L3a'];colors=dict(zip(groups,['black','#458b74','#999999','#4a80c0','#cc7b37','#ad4e50','#9171b5','#428d99']))
 for name in groups:
  rr=[summary[name]] if name in ['H0','R1','R2'] else [summary[f'{name}_s{seed}'] for seed in c['seeds']]
  x=[r['wrong_active_s'] for r in rr];y=[r['strict_hits'] for r in rr]
  axes[0].scatter(x,y,s=24,color=colors[name],alpha=.7);axes[0].scatter(np.mean(x),np.mean(y),marker='D',s=60,color=colors[name],label=name)
  if name in ['H0','L1b','L2','L3a']:axes[0].annotate('H0/R2' if name=='H0' else name,(np.mean(x),np.mean(y)),xytext=(5,-12 if name=='H0' else 6),textcoords='offset points',fontsize=9)
  if name in ['H0','R1','R2','L0','L1a']:axes[1].scatter(x,y,color=colors[name],s=45,label=name)
 axes[0].set(xlabel='Wrong Active seconds (lower better)',ylabel='Strict correct commits /54',title='All policies: points=seeds, diamonds=means');axes[0].legend(fontsize=8,ncol=2)
 axes[1].set(xlim=(70,105),ylim=(19.5,26),xlabel='Wrong Active seconds',ylabel='Strict correct commits /54',title='Close-up: H0 / rules / candidate gates');axes[1].legend(fontsize=8)
 fig.tight_layout();fig.savefig(out/'plots/val_active_tradeoffs.png',dpi=170);plt.close(fig)
 # Worst-source casebook by absolute Wrong Active increase over H0.
 base={r['episode_id']:r for r in summary['H0']['by_source']};cases=[]
 for name,r in summary.items():
  if name=='H0' or name.endswith('_mass'):continue
  for e in r['by_source']:cases.append((e['wrong_active_s']-base[e['episode_id']]['wrong_active_s'],name,e['episode_id']))
 ranked=sorted(cases,reverse=True);cases=[];seen=set()
 for case in ranked:
  if case[2] not in seen:cases.append(case);seen.add(case[2])
  if len(cases)==4:break
 fig,axes=plt.subplots(len(cases),1,figsize=(12,8));colors=dict(zip(LABELS,['#538eca','#409967','#d89f35','#956ab2']))
 metadata={e['episode_id']:e for e in read(Path(c['baseline_root'])/'joint/manifest.json')['episodes']}
 for ax,(delta,name,eid) in zip(axes,cases):
  e=metadata[eid];rr=read(out/'replays'/name/'val'/(eid+'.json'))['replay'];baseline=read(out/'replays/H0/val'/(eid+'.json'))['replay']
  for y,segs in [(2,[dict(start_s=s['start_frame']/e['fps'],end_s=s['end_frame']/e['fps'],stage=s['label']) for s in e['annotation']['segments']]),(1,baseline['active_intervals']),(0,rr['active_intervals'])]:
   for s in segs:ax.barh(y,s['end_s']-s['start_s'],left=s['start_s'],color=colors[s['stage']],height=.65)
  ax.set_yticks([0,1,2],[name,'H0','GT']);ax.set_title(f'{eid}: Wrong Active delta {delta:+.2f}s',fontsize=9)
 from matplotlib.patches import Patch
 fig.legend(handles=[Patch(color=color,label=stage) for stage,color in colors.items()],loc='upper center',ncol=4)
 fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'plots/near_boundary_casebook.png',dpi=160);plt.close(fig)
