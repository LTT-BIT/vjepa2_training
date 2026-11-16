"""Generate reviewable offline analysis and lightweight handoff; never fit/select."""
import gzip,json,zipfile,platform
from collections import Counter
import numpy as np
from .analyze import temporal_strata
from .contract import *
from .score import csv_rows


def attribution(out,summary):
 rows=[];transitions=[];decomposed={};cases={};cohorts={}
 base={r['episode_id']:r for r in summary['H0']['by_source']}
 for name,s in summary.items():
  reason_counts=Counter();stage_counts=Counter();by_source=[];cohorts[name]={}
  for src in s['by_source']:
   eid=src['episode_id'];raw=read(out/'replays'/name/'val'/(eid+'.json'))['replay'];events=raw['events'];bytime={}
   for ev in events:bytime.setdefault(ev['time'],[]).append(ev)
   commits=[ev for ev in events if ev['kind']=='commit']
   with gzip.open(out/'decisions'/name/'val'/(eid+'.jsonl.gz'),'rt') as f:
    for line in f:
     rec=json.loads(line);c=rec['candidate'];ft=rec['feature'];d=rec['decision'];now=ft['available_at'];same=bytime.get(now,[]);requested=bool(d and d['action']=='REQUEST');blocked=rec['blocked_reason'];key='no_candidate' if c['stage'] is None else 'shell_'+str(blocked) if blocked else 'gate_REQUEST' if requested else 'gate_HOLD' if d else 'H0_default_rule';reason_counts[key]+=1
     if requested and blocked is not None:reason_counts['REQUEST_rejected']+=1
     ce=next((x for x in commits if x['time']>=now and x['evidence']['stage']==c['stage'] and x['evidence']['first_available_at']<=now<=x['time']),None);related=[ev for ev in events if ce and ev['prompt_version']==ce['prompt_version'] and ev['time']>=ce['time']];ack=next((ev for ev in related if ev['kind'] in ['ack','late_ack']),None)
     rows.append(dict(strategy=name,episode_id=eid,observation_id=ft['observation_id'],policy_decision_at=now,first_observed_at=c['first_observed_at'],first_available_at=c['first_available_at'],candidate_stage=c['stage'],candidate_confidence=c['confidence'],gate_score=d['score'] if d else None,gate_REQUEST=requested,request_validated=requested and blocked is None,request_event_at=next((ev['time'] for ev in same if ev['kind']=='request'),None),blocked_reason=blocked,planning_tick=ft['commit_tick'],request_at=ce['evidence']['request_at'] if ce else None,commit_at=ce['time'] if ce else None,ack_at=ack['time'] if ack else None,late_ack=bool(ack and ack['kind']=='late_ack'),hard_timeout_at=next((ev['time'] for ev in related if ev['kind']=='hard_timeout'),None),prompt_version=ce['prompt_version'] if ce else None,attribution=key))
   counts=Counter(e['kind'] for e in events);stage_counts.update(dict(request_events=counts['request'],actual_commits=counts['commit'],normal_acks=counts['ack'],late_acks=counts['late_ack'],hard_timeouts=counts['hard_timeout']))
   for t in src['transitions']:
    trans=dict(strategy=name,episode_id=eid,**t);transitions.append(trans)
    for key,keep in [(t['from_stage']+'->'+t['to_stage'],True),('short_return',t['short_return']),('final_return',t['final_return']),('return_'+str(t['return_occurrence']),t['to_stage']=='return')]:
     if not keep:continue
     a=cohorts[name].setdefault(key,dict(total=0,matched=0,strict=0,early=0,late=0));a['total']+=1;a['matched']+=t['matched'];delta=t['commit_delta_s'];a['strict']+=delta is not None and -.2<=delta<=.5;a['early']+=delta is not None and delta<=-.5;a['late']+=delta is not None and delta>.5
   b=base[eid];by_source.append(dict(episode_id=eid,source_recording_id=src['source_recording_id'],strict_delta=src['strict_hits']-b['strict_hits'],wrong_delta_s=src['wrong_active_s']-b['wrong_active_s'],early_delta=src['early_ge05']-b['early_ge05'],hard_delta=src['hard_timeouts']-b['hard_timeouts']))
  decomposed[name]=dict(anchor_reasons=dict(reason_counts),event_pipeline=dict(stage_counts),longest_blocked_s=max(r['longest_blocked_s'] for r in s['by_source']),wrong_s_mean=float(np.mean([r['wrong_active_s'] for r in s['by_source']])),wrong_s_median=float(np.median([r['wrong_active_s'] for r in s['by_source']])),worst_source=max(s['by_source'],key=lambda x:x['wrong_active_s'])['source_recording_id'])
  order=sorted(by_source,key=lambda x:(x['wrong_delta_s'],-x['strict_delta']));cases[name]=dict(all_sources=by_source,improved_top3=[r for r in order if r['wrong_delta_s']< -1e-7 or r['strict_delta']>0][:3],degraded_top3=[r for r in reversed(order) if r['wrong_delta_s']>1e-7 or r['strict_delta']<0][:3],unchanged_sources=sum(abs(r['wrong_delta_s'])<=1e-7 and r['strict_delta']==0 for r in by_source))
 p=out/'summary/candidate_request_commit_ack_attribution.jsonl.gz';p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(p,'wt') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 write(out/'summary/pipeline_decomposition.json',decomposed);write(out/'summary/casebook.json',cases);write(out/'summary/transition_cohorts.json',cohorts);csv_rows(out/'summary/transition_scores_all.csv',transitions)
 return decomposed,cases,cohorts


def generate(c,out):
 s=read(out/'summary/val_overall.json');sel=read(out/'selection.lock.json');hist=read(out/'summary/historical_test_locked.json');models=read(out/'model_index.json');audit=read(out/'train_labels_audit.json');cover=read(out/'candidate_coverage_oracle_diagnostic.json');shift=read(out/'teacher_shift_report.json');aggregation=read(out/'aggregation.lock.json');boot=read(out/'source_bootstrap.json');decomp,cases,cohorts=attribution(out,s);verify=read(out/'logs/verification.json');integrity=read(out/'logs/integrity.json');tests=read(out/'logs/tests.json');lat=read(out/'hardware_latency.json');stress=read(out/'stress/summary.json')
 rows=[]
 for name,r in s.items():rows.append(dict(strategy=name,**{k:r[k] for k in ['strict_hits','matched','missed','unmatched_commits','early_ge05','late_gt05','wrong_active_s','exact_order','hard_timeouts','recovery_required_s','ack_wait_s']}))
 csv_rows(out/'summary/val_by_seed.csv',rows)
 metadata={e['episode_id']:e for e in read(Path(c['baseline_root'])/'joint/manifest.json')['episodes']};replays={};decisions={}
 for name in s:
  replays[name]={p.stem:read(p)['replay'] for p in (out/'replays'/name/'val').glob('*.json')};decisions[name]={}
  for p in (out/'decisions'/name/'val').glob('*.jsonl.gz'):
   with gzip.open(p,'rt') as f:decisions[name][p.name.removesuffix('.jsonl.gz')]=[json.loads(line) for line in f]
 strata=temporal_strata(metadata,replays,decisions);write(out/'summary/temporal_strata.json',strata);del decisions,replays
 lines=['# 番茄同候选空间握手门控补充消融报告（2026-11-13）','',f'## 结论：{sel["selected"]}','',f'本轮实际完成 H0、R1 和 G0/G1/G2/G3×3 seed。验证集固定9来源/54转换；正式版本 `{aggregation["final_version"]}`。选型要求三seed分别满足全部风险门槛，代表seed固定42；没有根据test调参。默认握手机制保持 H0。','',f'运行目录：`{out}`。分支 `feat/tomato-candidate-gate-v3-20261113`，基线 `db4cdad6e87af7c077f5be16a00bc277efeb54d9`。本轮未提交、未推送。','', '## 1. 实验身份与受控差异','', '原LOCAL93清洗normal、source_recording_id分组75/9/9不变；单一官方B0 Joint seed42双头、+1秒/2Hz。训练仅复用5折OOF（每折54train/6内val/15held）概率，val/test使用全75训练官方B0。三seed只改变门控初始化/批次，不能当成27条验证录像。未调用Encoder，未重新提取视觉或训练Memory/K1。','', f'协议SHA：`{sha(out/"protocol.lock.json")}`。输入SHA、93来源、标注、模型/normalization、五折成员和概率逐文件身份分别见source_identity.json / split.lock.json / oof_teacher_manifest.json / protected_sha256.json。校验保护文件 {integrity["protected_files"]} 个，差异 {len(integrity["differences"])} 个。','', '修正审计：首个输出handshake_candidate_gate_v3_20261113_110116_4ee71f保留全部初跑结果，train尾部核验发现未知主gate标签误记HOLD负例，已显式标记SUPERSEDED。只修正主gate mask并对所有模型统一重跑，未改阈值/超参数/准入规则；正式结果以本运行目录为准。初跑仅H0进入历史test，没有失败学习器test筛选。','', '上一轮全量成果和源码均只读复用。本轮唯一生产代码变更：handshake.py提取receive_evidence（只处理ACK/TTL）与propose_default_candidate；原prediction仍顺序调用二者。新CandidateRequestPort只复用旧apply的请求安全检查，覆盖ingest并传入真实Future，从不调用旧neutralize路径。H0/R1完整事件/Active/Pending/summary与历史逐值一致（H0额外84条train/val instrumented-vs-pure）。原GUI/四分类训练/柠檬/π0.5/Agent/RTC未改。','', '统一候选：Future top1≠自身Active、外部图合法、p≥.65，至多一个；所有模型只输出HOLD/REQUEST（≥.5），不能输出另一目标。G0/G1/G3输入同一240维（4×59+4mask），G2读取同一4×59和mask；4锚点仅概率/自身在线状态，非视觉Memory。G3的6桶仅辅助监督，不参与动作。','', '## 2. 验证集最终 Active 主结果','', '|策略|窄窗/54|宽窗|漏切|未匹配|过早≤−.5s|迟到>.5s|Wrong(s)|完整顺序/9|硬超时|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n,r in s.items():lines.append(f"|{n}|{r['strict_hits']}|{r['matched']}|{r['missed']}|{r['unmatched_commits']}|{r['early_ge05']}|{r['late_gt05']}|{r['wrong_active_s']:.3f}|{r['exact_order']}|{r['hard_timeouts']}|")
 lines+=['','所有指标均为本轮真实冻结概率回放，不是合成测试精度。使用旧一对一from→to匹配器：宽窗[-1,+2]、严格[-.2,+.5]，边界保留帧时间。提前/迟到统计限定已宽窗匹配，额外提交另列；Wrong按完整Active和GT交叠积分。','', '### 三seed准入审计','', '|组|各seed通过全部风险|均值strict|均值Wrong(s)|最坏过早|最坏硬超时|达到收益门槛|最终准入|','|---|---|---:|---:|---:|---:|---|---|']
 for g,a in sel['audit'].items():lines.append(f"|{g}|{a['seed_passes']}|{a['mean_strict']:.2f}|{a['mean_wrong_s']:.3f}|{a['worst_early']}|{a['worst_hard']}|{a['benefit']}|{a['qualifies']}|")
 lines+=['','预锁：各seed illegal/cross-session/nonplanning均0，early≤4、hard≤3、miss≤11、extra≤11、strict≥23、exact order≥9、Wrong≤98.233333s；所有seed通过后，组均值strict≥25或Wrong≤88.410s才有收益。没有best-seed选择、没有放宽门槛。','', '## 3. 候选覆盖与候选→请求→提交→ACK归因','',f'GT-aligned Active的oracle候选诊断：{cover["oracle_covered"]}/{cover["events_total"]}边界在严格窗可获得合法新鲜Future候选，缺失{cover["events_total"]-cover["oracle_covered"]}个。H0自身状态下可执行覆盖 {cover["h0_executable_covered"]}，仅阻塞条件覆盖 {cover["h0_blocked_covered"]}（两集合可重叠）。oracle只衡量候选供给，不是在线输入；H0条件覆盖也不是其他策略的绝对上界。','', '|策略|无候选锚点|有候选HOLD|REQUEST被拒锚点|请求事件|实际commit|普通ACK|迟到ACK|硬超时|最长阻塞(s)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n in ['H0','R1']+[g+'_s42' for g in c['groups']]:
  d=decomp[n];a=d['anchor_reasons'];e=d['event_pipeline'];lines.append(f"|{n}|{a.get('no_candidate',0)}|{a.get('gate_HOLD',0)}|{a.get('REQUEST_rejected',0)}|{e['request_events']}|{e['actual_commits']}|{e['normal_acks']}|{e['late_acks']}|{e['hard_timeouts']}|{d['longest_blocked_s']:.3f}|")
 lines+=['','此表“被拒锚点”仅统计真正REQUEST且被壳拒绝，ACK阻塞下的日志HOLD另记；精确REQUEST/validated/request-event/commit/ACK关联保存在summary/candidate_request_commit_ack_attribution.jsonl.gz。候选提供时刻（first_observed/available）与壳内Pending开始不同，均记录，不能把网络REQUEST直接算成功。未决请求、ACK阻塞、错误预测、尾部删失分开。','', '## 4. 监督、统一纠偏与实际训练','', '门控只在新鲜合法候选且可请求状态形成样本，以下一实际planner tick标定。严格窗主正例；若GT当前仍是候选且前一真值类是自身Active，则保留相邻late-catchup正例（不能据此复位Active），其他错配HOLD。每来源/事件/占用时间桶等权，train加权neg/pos（上限10）；故输出是重加权研究分数，未经自然先验校准，不是物理安全概率。','', 'G0 BCE+.5*.01*||非biasW||²，AdamW decay0；其他decay.01。G3 BCE+.25×有效hazard样本加权平均censored NLL；右删失将兼容桶概率求和，不把末尾当无转换负例。6桶delta=b−tick：<−.5、[−.5,0]、(0,.2]、(.2,.5]、(.5,1]、>1，边界浮点按1e−9处理。无法确定候选事件的hazard mask，不伪造边界/FINISH。尾部若end−tick<.2秒，下一真实边界是否在正窗内未知，主gate BCE也mask（hazard仍按右删失学习）。','', f'H0-OOF初始训练可执行tick {audit["train"]["eligible_ticks"]}（其中主gate未知屏蔽{audit["train"]["gate_censored_masked"]}个），正{audit["train"]["positives"]}/负{audit["train"]["negatives"]}，覆盖来源 {audit["train"]["source_count"]}/75；observed {audit["train"]["observed_boundaries"]}、censored {audit["train"]["censored"]}、hazard masked {audit["train"]["hazard_masked"]}。无可执行候选来源仍完整回放，未删除。每来源/转换/桶数量、最多21例确定性边界审计与±1帧/±.1s标签敏感性见train_labels_audit.json，未修改标注。','', f'全局纠偏触发：{aggregation["triggered"]}。先完成全部12个v0，仅train OOF比较自身Active累积偏移>10s或未见状态组合>5%；任一触发则所有12个一律聚合一次、重初始化并使用v1。每个v1使用相同baseN+ownN样本预算（不足有放回并留SHA）、相同种子批次顺序；共享统计只池化train base和全部12个own来源。早停统一val-H0监督gate BCE，最多80epoch/patience10，不看val Active选epoch/v0v1。实际更新数因预定早停不同，未额外扩预算。','', '|模型|参数|best/实际epoch|updates|train tick|正/负|训练(s)|GPU峰值MiB|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for n,m in models.items():
  a=m['manifest'];t=a['train_audit'];lines.append(f"|{n}|{a['parameters']}|{a['best_epoch']}/{a['epochs']}|{a['updates']}|{t['eligible_ticks']}|{t['positives']}/{t['negatives']}|{a['training_seconds']:.2f}|{a['peak_gpu_allocated_bytes']/2**20:.2f}|")
 lines+=['','### 门控与辅助头（最终自身轨迹，仅可请求tick）','', '|模型|正例/n|PR-AUC|Precision|Recall|Brier|ECE|','|---|---:|---:|---:|---:|---:|---:|']
 for n in models:
  gm=read(out/'evaluations'/n/'val/gate_metrics.json')['gate'];lines.append(f"|{n}|{gm.get('positives',0)}/{gm['n']}|{gm.get('pr_auc',0):.3f}|{gm.get('precision',0):.3f}|{gm.get('recall',0):.3f}|{gm.get('brier',0):.3f}|{gm.get('ece',0):.3f}|")
 lines+=['','各组自身轨迹的可请求集合不同，因此门控分类分数不宜脱离Active横排。val监督早停集合相同；训练log保留BCE/hazard两项。G3各seed的观察桶confusion、bin-MAE、Brier/ECE和censor-compatible NLL见对应gate_metrics.json；尾部桶无界，未伪造连续秒MAE。','', '## 5. OOF→full-B0分布漂移','', '|概率来源|Current准确率/ECE|Future准确率/ECE|双头分歧|候选比例|可请求比例|','|---|---:|---:|---:|---:|---:|']
 for sp in ['train','val']:
  a=shift[sp];lines.append(f"|{a['teacher']}|{a['heads']['current']['accuracy']:.3f}/{a['heads']['current']['ece']:.3f}|{a['heads']['future']['accuracy']:.3f}/{a['heads']['future']['ece']:.3f}|{a['current_future_disagreement']:.3f}|{a['candidate_fraction']:.3f}|{a['executable_fraction']:.3f}|")
 lines+=['','概率分位数、边界±1秒质量、NLL/Brier及fold身份详见teacher_shift_report.json。来源数量不同且val仅9条，描述差异不能唯一归因为teacher漂移；没有在val拟合校准器。','', '## 6. 配对统计、正常样本回归与边界类型','', '来源为bootstrap单位，5000次固定rng20261113；seed先在每来源平均，绝不将seed当独立录像。区间为总量差，对反复研究的val只作探索性描述。','', '|差值(后者−前者)|strict总量差 [95%CI]|Wrong秒差 [95%CI]|','|---|---:|---:|']
 for k in [g+'_vs_H0_seedmean' for g in c['groups']]+['G1_minus_G0_sourcepaired_seedmean','G2_minus_G1_sourcepaired_seedmean','G3_minus_G1_sourcepaired_seedmean']:
  a=boot[k]['strict_hits'];b=boot[k]['wrong_active_s'];lines.append(f"|{k}|{a['delta_total']:.2f} [{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}]|{b['delta_total']:.3f} [{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}]|")
 for g in c['groups']:
  n=g+'_s42';v=cases[n];lines+=['',f'**{n}正常来源对照H0**：未变{v["unchanged_sources"]}/9；按Wrong秒排序（并列看strict），改善/退化各至多3例：']
  for kind in ['improved_top3','degraded_top3']:
   lines.append(f'- {kind}: '+('; '.join(f"`{x['episode_id']}` strictΔ{x['strict_delta']:+d} / WrongΔ{x['wrong_delta_s']:+.3f}s / earlyΔ{x['early_delta']:+d} / hardΔ{x['hard_delta']:+d}" for x in v[kind]) or '无'))
 lines+=['','所有9来源、全部3seed包括最坏样本均保留在casebook与episode_scores；不能只举原问题样本。','', '|策略seed42|harvest→return strict/总|return→adjust|adjust→pickup|末次return|短return≤1.5s|','|---|---:|---:|---:|---:|---:|']
 for n in ['H0','R1']+[g+'_s42' for g in c['groups']]:
  row=[]
  for key in ['harvest->return','return->adjust','adjust->pickup','final_return','short_return']:
   a=cohorts[n].get(key,dict(strict=0,total=0));row.append(f"{a['strict']}/{a['total']}")
  lines.append('|'+n+'|'+'|'.join(row)+'|')
 lines+=['','|策略|边界±.5s Wrong/时长(s)|边界±1s|稳定>1s|','|---|---:|---:|---:|']
 for name in ['H0','R1']+[g+'_s42' for g in c['groups']]:
  values=[strata[name]['active_time'][k] for k in ['boundary_pm05','boundary_pm1','stable_gt1']];lines.append('|'+name+'|'+'|'.join(f"{v['wrong_active_s']:.3f}/{v['duration_s']:.3f}" for v in values)+'|')
 lines+=['','以上窗口是时间并集，±.5包含于±1；稳定>1与±1互补，不将重叠窗口重复计总Wrong。Pending/ACK状态时长与延迟窗口Brier见temporal_strata.json。','', '重复return按发生次序分列，短段与末尾分列：transition_cohorts.json。最终return之后没有已观察边界时不能制造return→adjust/FINISH；当前概率预测与已发生的物理完成不是同一监督。','', '## 7. 固定故障压力测试','', 'H0/R1/G0–G3 seed42均完成13种×9来源=702条注入回放。注入仅按源锚点索引模运算，或因果首次commit后5.5s无更新，不按GT边界挑时机；+0.1/+0.3s、长gap、重复、倒序、跨session/model、HELD、单/连续错Future、Current翻转/丢失、直到硬超时均覆盖。','', '|注入|H0 strict/Wrong/hard|R1|G0|G1|G2|G3|','|---|---|---|---|---|---|---|']
 for scenario,v in stress.items():lines.append('|'+scenario+'|'+'|'.join(f"{v[n]['strict_hits']}/{v[n]['wrong_active_s']:.1f}/{v[n]['hard_timeouts']}" for n in ['H0','R1','G0','G1','G2','G3'])+'|')
 lines+=['','非法/跨会话/非规划commit均0，全部版本单调且TTL受限。错误但格式合法的概率可能改变合法提交，协议无法自动识别语义错类；因此上表不是“故障下性能不变”。所有记录见stress/，不用于选型。','', '## 8. 锁定后的 historical test','', f'先写selection/test_lock，再只运行H0及唯一winner；本轮实际评估 {list(hist)}。该9来源历史已反复曝光，不是新盲测；没有用它搜索模型或阈值。','', '|策略|窄窗/54|漏切|额外|过早|Wrong(s)|硬超时|','|---|---:|---:|---:|---:|---:|---:|']
 for n,r in hist.items():lines.append(f"|{n}|{r['strict_hits']}|{r['missed']}|{r['unmatched_commits']}|{r['early_ge05']}|{r['wrong_active_s']:.3f}|{r['hard_timeouts']}|")
 lines+=['','## 9. 验收与运行成本','',f'实际pytest：**{tests["result"]}**，完整输出logs/pytest.log。模型重新加载、14策略×val9完整事件/分数重算一致，batch-vs-stream误差≤{max(verify["batch_stream_max_absolute_error"].values()):.3g}，删除GT字段推理输入/决策完全一致。共审计{verify["replay_files"]}个回放、{verify["actual_commits"]}次实际commit协议约束。','', '|模型|CPU前向P50/P95(ms)|GPU前向P50/P95(ms)|','|---|---:|---:|']
 for n,a in lat.items():lines.append(f"|{n}|{a['cpu']['batch1_ms_p50']:.4f}/{a['cpu']['batch1_ms_p95']:.4f}|{a['cuda']['batch1_ms_p50']:.4f}/{a['cuda']['batch1_ms_p95']:.4f}|")
 lines+=['','batch1纯决策网络、200次测量、预热20、单CPU线程，GPU同步；不包含Encoder/视频/π0.5并发。4锚点输入占944+4字节有效数值缓存，另有Python/状态开销；每episode重新创建特征/候选/执行状态，GRU不跨4锚点保存hidden。','', '## 10. 文件与复现','', '新增：configs/tomato_candidate_gate_v3_20261113.yaml；tools/tomato_review_ui/candidate_gate_v3/*.py及README；tests/test_tomato_candidate_gate_v3.py；本报告和docs/validation/tomato_candidate_gate_v3_20261113。生产文件仅handshake.py兼容接口提取，原始文件副本original_handshake.py和before/after SHA保留。所有旧实验未覆盖。','', '```bash','PY=.venv-tomato/bin/python','EXP=runs/tomato/handshake_candidate_gate_v3_NEW_UNIQUE','$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --check-only','$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --smoke','$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --run','$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --verify','$PY -m tools.tomato_review_ui.candidate_gate_v3.cli --output-dir "$EXP" --report-only','$PY -m pytest -q tests/test_*tomato*.py tests/test_lemon_vjepa_phase.py','```','', '输出目录拒绝覆盖。完整权重、训练曲线、归一化、所有决策序列/事件、逐来源/逐转换CSV、bootstrap/压力测试均在run目录。handoff.zip仅报告、代码、配置、审计/汇总/训练日志/失败个例，不含视频、原始特征、Encoder或完整模型权重。','', '## 11. 最终研究意见与未验证项','']
 for g in c['groups']:
  a=sel['audit'][g];lines.append(f'- {g}：三seed strict均值{a["mean_strict"]:.2f}、Wrong均值{a["mean_wrong_s"]:.3f}s；准入{a["qualifies"]}。风险失败与平均收益应分开看，不能拿更低BCE取代最终Active。')
 lines+=['', '本轮只支持“同候选/同监督/同历史”的局部结构对照；旧五动作GRU/Hazard退化不能直接解释为结构无效，同样也不能从9条val证明新结构普适有效。候选缺口、OOF到full-B0概率漂移、粗边界监督、ACK阻塞都需要分别处理。','', '若未通过全部准入，建议继续保留H0，将后续精力放在候选供给、边界监督与独立实时Shadow；不在本轮结果上继续搜阈值/模型。即使通过也只作为Shadow研究候选。K1敏感性NOT_RUN（可选，非主线）；新独立录像、人工逐帧物理完成核验、真机闭环、Encoder+π0.5并发、生产延迟/成功率均未验证。']
 report='\n'.join(lines)+'\n';dest=REPO/'docs/TOMATO_HANDSHAKE_CANDIDATE_GATE_V3_REPORT_20261113.md';dest.write_text(report);(out/'REPORT.md').write_text(report)
 # Scientific static summary; no fabricated performance points.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4));groups=c['groups']
 for i,g in enumerate(groups):
  for j,key in enumerate(['strict_hits','wrong_active_s']):ax[j].scatter([i]*3,[s[f'{g}_s{seed}'][key] for seed in c['seeds']],label=g if j==0 else None)
 for j,(key,title) in enumerate([('strict_hits','Strict hits (54 boundaries)'),('wrong_active_s','Wrong Active (seconds)')]):
  ax[j].axhline(s['H0'][key],color='black',ls='--',label='H0');ax[j].set_xticks(range(4),groups);ax[j].set_title(title);ax[j].grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'summary/val_seed_scatter.png',dpi=160);plt.close(fig)
 validation=REPO/'docs/validation/tomato_candidate_gate_v3_20261113';validation.mkdir(parents=True,exist_ok=True)
 for src in [out/'logs/verification.json',out/'logs/integrity.json',out/'logs/tests.json',out/'selection.lock.json']:write(validation/src.name,read(src))
 files=[p for p in out.rglob('*') if p.is_file() and p.name not in ['handoff.zip','artifacts_sha256.json']];write(out/'artifacts_sha256.json',{str(p.relative_to(out)):sha(p) for p in sorted(files)})
 with zipfile.ZipFile(out/'handoff.zip','w',zipfile.ZIP_DEFLATED) as z:
  z.write(dest,dest.name)
  for p in Path(__file__).parent.glob('*.py'):z.write(p,'code/'+p.name)
  for p in [REPO/'configs/tomato_candidate_gate_v3_20261113.yaml',REPO/'tests/test_tomato_candidate_gate_v3.py']:z.write(p,'code/'+p.name)
  for p in out.rglob('*'):
   if not p.is_file():continue
   rel=p.relative_to(out)
   if p.suffix in ['.pt','.npz'] or rel.parts[0] in ['decisions','replays','smoke'] or p.name=='handoff.zip':continue
   z.write(p,str(rel))
  for group in c['groups']:
   for case in cases[group+'_s42']['degraded_top3'][:1]:
    p=out/'replays'/(group+'_s42')/'val'/(case['episode_id']+'.json');z.write(p,'failure_examples/'+group+'_'+p.name)
 print('REPORT',dest,'HANDOFF',out/'handoff.zip',flush=True)
