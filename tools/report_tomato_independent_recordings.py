"""Publish one-shot independent-test summaries without changing selection rules."""
from pathlib import Path
from collections import Counter
import numpy as np
from . import tomato_dataset as ds
ROOT=Path('runs/tomato/independent10_20261109')


def run():
    results=ds.read_json(ROOT/'results.json');p=ds.read_json(ROOT/'locked_protocol.json');m=ds.read_json(ROOT/'manifest.json');inv=ds.read_json(ROOT/'remote_inventory.json');rec={r['source_recording_id']:r for r in inv['records']};primary=results[p['primary']];hazard=ds.read_json(ROOT/'time_head_diagnostic.json');uncertainty=ds.read_json(ROOT/'uncertainty.json');summary={};misses=[];false=[]
    for name,r in results.items():
        summary[name]={k:v for k,v in r.items() if k!='per_episode'}
        summary[name]['common_1s_future_hits']=sum(v['future']['common_1s_hits'] for v in r['per_episode'].values())
        summary[name]['per_episode']={eid:{h:dict(classification=v[h]['classification'],total=len(v[h]['boundaries']),detected=sum(b['detected'] for b in v[h]['boundaries']),false_events=len(v[h]['false_events']),offsets_s=[b['offset_s'] for b in v[h]['boundaries'] if b['detected']]) for h in ['current','future']} for eid,v in r['per_episode'].items()}
    for e in m['episodes']:
        eid=e['episode_id'];v=primary['per_episode'][eid]['future']
        for ordinal,b in enumerate(v['boundaries'],1):
            if not b['detected']:
                same=[x for x in v['events'] if x['previous']==b['previous'] and x['label']==b['label']];nearest=min(same,key=lambda x:abs(x['time']-b['time'])) if same else None
                offset=nearest['time']-b['time'] if nearest else None
                misses.append(dict(episode_id=eid,package=rec[e['source_recording_id']]['package'],episode_index=e['episode_index'],ordinal=ordinal,transition=ds.LABELS[b['previous']]+'->'+ds.LABELS[b['label']],boundary_s=b['time'],nearest_same_transition_offset_s=offset,timing_evidence='no_same_class_event' if offset is None else ('too_early' if offset<-1.5 else ('late' if offset>=0 else 'one_to_one_conflict'))))
        for event in v['false_events']:
            same=[b for b in v['boundaries'] if b['previous']==event['previous'] and b['label']==event['label']];nearest=min(same,key=lambda b:abs(b['time']-event['time'])) if same else None;offset=event['time']-nearest['time'] if nearest else None
            false.append(dict(episode_id=eid,**event,offset_s=offset,timing_evidence='wrong_transition_type' if offset is None else ('too_early' if offset<-1.5 else ('late' if offset>=0 else 'duplicate_or_conflict'))))
    seedstats={k:[summary[f'independent_h15_s{s}']['events']['future'][k] for s in [42,43,44]] for k in ['recall','false_per_minute']}
    metadata=dict(locked_before_scoring=True,new_recordings=10,original_annotation_events=primary['events']['future']['total'],old_source_overlap=False,training_calls=0,recommended_model_unchanged=True,new_model_selected=False,seed_summary={k:dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1))) for k,v in seedstats.items()},primary_recording_ci=uncertainty,primary_miss_diagnoses=dict(Counter(x['timing_evidence'] for x in misses)),primary_false_diagnoses=dict(Counter(x['timing_evidence'] for x in false)),time_head_diagnostic=hazard['summaries'])
    doc=Path('docs/validation/tomato_independent10_20261109');doc.mkdir(exist_ok=False)
    ds.write_json(doc/'model_metrics.json',summary);ds.write_json(doc/'summary.json',metadata);ds.write_json(doc/'missed_events.json',misses);ds.write_json(doc/'false_events.json',false)
    for name in ['locked_protocol.json','remote_inventory.json','download_receipt.json','validation.json','integrity.json']:
        ds.write_json(doc/name,ds.read_json(ROOT/name))
    lines=['# 新上传10条番茄录制：冻结模型独立评估（2026-11-09）','','## 测试身份与边界','','数据来自28服务器的A/1109_783_818_normal（episode1—7）和B/1109_696_811_normal（episode1—3）。包内两个未标注编号0条目未纳入。10条均有已保存、reviewed且非inherited的七段粗标注。主测前锁定模型SHA、阈值0.5/K2、2Hz和匹配规则；新数据全部只用于评估，没有训练、选epoch、改阈值、重拟合归一化或选择新模型。','','按source_recording_id与旧94条全量对照无交集；全部选中Parquet与视频SHA和旧数据、彼此不重复。远端和本地每文件SHA一致。三路完整解码的帧数/PTS、58D按名称对齐、因果16帧×10Hz历史均核验。粗阶段标注不等于逐条物理采摘成功已人工复核；没有凭模型分数排除新测试样本。','','这是此前未用于训练/验证的录制，仍可能共享场景/操作者。只能支持未知录制泛化，不等于跨场景/跨机器人的泛化。现在评分已揭示，这批数据若以后用于调参就不再是新的盲测；本轮不反复搜索。','','## 固定模型结果','','|模型|Current Macro-F1|Future Macro-F1|Current事件|Future事件|Future Precision|Future误报/分钟|Future中位提前秒|共同1秒窗命中|','|---|---:|---:|---|---|---:|---:|---:|---:|']
    for name,r in summary.items():
        f=r['events']['future'];c=r['events']['current'];lines.append(f"|{name}|{r['classification']['current']['macro_f1']:.5f}|{r['classification']['future']['macro_f1']:.5f}|{c['detected']}/{c['total']}|{f['detected']}/{f['total']}|{f['precision']:.3f}|{f['false_per_minute']:.3f}|{-f['offset_p50_s'] if f['offset_p50_s'] is not None else float('nan'):.3f}|{r['common_1s_future_hits']}|")
    lines+=['','不同Future时距分别评估固定1秒/1.5秒目标，不能直接按Future F1排名；共同1秒事件窗作为额外公平对照。Current事件指分类流确认，不冒充七阶段Agent/Prompt实际确认。所有模型保留原训练统计；没有用新测试进行归一化适配。','','## 主测逐录制','','|原数据|Current F1|Future F1|预警命中|未匹配预警|','|---|---:|---:|---|---:|']
    for e in m['episodes']:
        eid=e['episode_id'];v=summary[p['primary']]['per_episode'][eid];src=rec[e['source_recording_id']];f=v['future'];lines.append(f"|{src['package']}/episode_{e['episode_index']:06d}|{v['current']['classification']['macro_f1']:.4f}|{f['classification']['macro_f1']:.4f}|{f['detected']}/{f['total']}|{f['false_events']}|")
    lines+=['','## 转换类型与不确定性','',str(primary['events']['future']['by_transition']),'',str(metadata['seed_summary']),'',str(uncertainty),'','主测漏检数值归因：'+str(metadata['primary_miss_diagnoses'])+'；未匹配预警归因：'+str(metadata['primary_false_diagnoses'])+'。这是时序证据，不凭它推断视觉遮挡或篡改人工边界。逐事件清单见missed_events.json/false_events.json。','','## 新时间头固定诊断','',str(hazard['summaries']),'','这是第四阶段已失败模型的预先锁定阈值0.5诊断，未在新数据重选阈值。分类分数不变，事件决策另计；不因测试集某个均值更高就替换现有模型。','','## 产物','','原始数据仅下载到/home/user/datasets/tomato_independent10_20261109，不回写服务器。完整manifest、独立新特征缓存、五组模型真实logits、协议与结果位于'+str(ROOT.resolve())+'。版本化文件仅包含代码、摘要和可核对身份，不上传原数据/权重。没有操作机器人或修改OpenPI/RTC。本轮沿用不推送远程的用户约束。','']
    Path('docs/TOMATO_INDEPENDENT10_RESULTS_20261109.md').write_text('\n'.join(lines))
    print('REPORT COMPLETE',primary['events']['future'],metadata['seed_summary'])

if __name__=='__main__':run()
