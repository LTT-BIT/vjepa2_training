"""Publish clean-data refresh results; exposed new10 is never scored as a blind test."""
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds
from .tomato_independent_heads import load_checkpoint
ROOT=Path('runs/tomato/refresh101_20261109_v2')


def run():
    data=ds.read_json(ROOT/'summary.json');records=data['results'];rng=np.random.default_rng(42);paired=[];ids=list(records[0]['evaluation']['per_episode'])
    for _ in range(3000):
        sample=[ids[i] for i in rng.integers(0,len(ids),len(ids))];diff=[]
        for seed in [42,43,44]:
            a=next(r for r in records if r['seed']==seed and r['cohort']=='clean91_control')['evaluation']['per_episode'];b=next(r for r in records if r['seed']==seed and r['cohort']=='refresh101')['evaluation']['per_episode'];n=sum(len(a[e]['future']['boundaries']) for e in sample);delta=sum(sum(x['detected'] for x in b[e]['future']['boundaries'])-sum(x['detected'] for x in a[e]['future']['boundaries']) for e in sample)/n;diff.append(delta)
        paired.append(np.mean(diff))
    compact=[]
    for r in records:
        saved,_=load_checkpoint(r['bundle'],'cpu');assert ds.digest(r['bundle'])==r['bundle_sha256'];cf=r['evaluation'];training=[]
        for mode in ['current_only','future_only']:
            result=ds.read_json(ROOT/r['cohort']/f'{mode}_s{r["seed"]}'/'result.json');training.append({k:v for k,v in result.items() if k!='history'})
        compact.append(dict(cohort=r['cohort'],seed=r['seed'],bundle=r['bundle'],bundle_sha256=r['bundle_sha256'],classification=cf['classification'],events=cf['events'],training=training,per_episode=cf['per_episode'],parameter_count=2050568))
    output=dict(results=compact,aggregate=data['aggregate'],gate_passed=data['new_research_candidate_gate_passed'],paired_recall_delta_ci95=np.percentile(paired,[2.5,97.5]).tolist(),interpretation='No blind-test evaluation; new10 has been explicitly promoted to training after its frozen independent result',old_B0_kept=True,stable_default_unchanged=True)
    doc=Path('docs/validation/tomato_refresh101_20261109');doc.mkdir(exist_ok=False)
    ds.write_json(doc/'results.json',output)
    for name in ['data_identity.json','protocol.json']:ds.write_json(doc/name,ds.read_json(ROOT/name))
    for cohort in ['clean91_control','refresh101']:
        mf=ds.read_json(ROOT/cohort/'manifest.json');ds.write_json(doc/(cohort+'_manifest_identity.json'),dict(manifest_path=str(ROOT/cohort/'manifest.json'),sha256=ds.digest(ROOT/cohort/'manifest.json'),split_sha256=mf['split_sha256'],counts={s:len(v) for s,v in mf['splits'].items()},label_version=mf['label_version']))
    cfg=dict(enabled_by_default=False,recommended=False,reason='Fixed validation event false-rate gate failed; do not replace original B0 or OpenPI default',checkpoint=str(ROOT/'refresh101/independent_h15_s42.pt'),model_version=ds.digest(ROOT/'refresh101/independent_h15_s42.pt'),runtime_class='tools.tomato_agent_inference.TomatoAgent',pretrained_dir='pretrained/vjepa2-vitl-fpc64-256',prediction_hz=2,future_horizon_s=1.5,current_threshold=.5,future_threshold=.5,confirmations=2,transition_time_checkpoint=None,split_role='84train9dev8archived_history_test',new_independent_test_available=False)
    ds.write_json('configs/tomato_refresh101_20261109/inference_experimental.json',cfg)
    lines=['# 101条数据版本：新一轮独立双头训练与清洁控制（2026-11-09）','','## 结论','','已实际完成12个GPU分类头训练（2组数据×3种子×Current/Future），全部沿用冻结视觉编码器及原MLP架构。新增10条后平均事件召回仅73.3%→74.0%，误报0.934→1.055次/分钟，未通过预设门槛，不替换B0或OpenPI默认。固定seed42新模型37/50、17误报，原B0为39/50、13误报。','','## 101条身份与新10条的用途变化','','服务器A/B恰101条有标注、唯一source_recording_id。旧94剔除整包3条后91条，加新10条=101条。远端所有视频/Parquet SHA与已有本地缓存源身份一致；标注只发现已批准的B/1106_750_791_normal/episode_000004修订，重建监督并保留旧版本。训练前服务器inventory原件保存在time_usage_analysis运行目录，摘要及SHA已锁定。','','clean91_control：74train/9dev/8历史test；refresh101：84train/9dev/8历史test。新10条在8032f2e已完成独立测试，本轮用户明确授权加入训练，现身份为开发/训练，绝不再拿它们训练后得分宣称盲测。旧8条仅保留元数据身份，未读取其特征数组或推理。后续需要另录封存新盲测。','','## 训练控制','','网络、16帧×10Hz、2Hz预测、未来1.5秒、三个种子42/43/44、100epoch上限/patience15、AdamW3e-4/weight_decay0.01、batch128、dropout0.3、sqrt逆频率CE和梯度裁剪1均保持不变。各头按自身开发验证Macro-F1再NLL选epoch，延续原独立头规则；没有按新10条的测试指标或门控阈值挑选。每个数据组只在各自训练录制拟合4000D均值/标准差，两头统计精确共享。没有重新提取视频特征或重训失败的时间头。所有12个配置都是此前未完成的数据版本训练，非重跑原31个配置。','','9条开发验证仍采用与39/50一致的14项部分人工边界提案、50事件、固定0.5/K2、严格同类一对一未来[b-1.5,b)和共同1秒窗口。原始标签与提案未混称正式批准版本。','','|组别|Seed|Current F1|Future1.5 F1|预警命中|Precision|事件F1|误报/分钟|adjust→pickup|训练秒|','|---|---|---:|---:|---|---:|---:|---:|---|---:|']
    for r in compact:
        q=r['events']['future'];a=q['by_transition']['adjust->pickup'];lines.append(f"|{r['cohort']}|{r['seed']}|{r['classification']['current']['macro_f1']:.5f}|{r['classification']['future']['macro_f1']:.5f}|{q['detected']}/50|{q['precision']:.3f}|{q['f1']:.3f}|{q['false_per_minute']:.3f}|{a['detected']}/{a['total']}|{sum(t['training_s'] for t in r['training']):.1f}|")
    lines+=['','## 方差与录制不确定性','',str(data['aggregate']),'',f"配对9录制bootstrap的三种子平均召回差95%区间：{output['paired_recall_delta_ci95']}。仅开发集探索区间，不是泛化保证。新组跨3次运行总命中只多1个（110→111/150；重复种子并非150个独立物理事件），误报反而增加；不能解释为可靠改善。",'','Current分类均值可能上升，但Future分类、事件F1和误报没有一致改善；adjust→pickup也不稳定（控制4/3/3，新组2/4/3）。没有证据因“数据更多”就自动改推荐。逐Episode、每类Precision/Recall、混淆矩阵与事件时间都在results.json。','','## 版本、checkpoint与边界','','所有原checkpoint和实验保留；新每头含best_head.pt、last_state.pt（权重/optimizer/RNG）、完整history、归一化和manifest，组合包同样包含两头与输入Schema。代表新模型为'+str((ROOT/'refresh101/independent_h15_s42.pt').resolve())+'，仅实验用途；控制包位于clean91_control同名位置。版本化推理配置为configs/tomato_refresh101_20261109/inference_experimental.json，默认禁用。完整新数据身份与监督SHA位于data_identity.json。','','首个准备尝试因targets已含anchors而重复传关键字，在任何GPU训练前失败；已修复，初始失败日志与目录保留，成功实验写入独立_v2目录。没有删除旧结果或为了通过校验改标签。','','## 下一步','','优先把细粒度边界规范与数据成功/失败质量做一致复核，保留时间头作为非否决辅助信息；离线研究阶段确认失败时的保守恢复与“不确定”状态，防止前序失配使后续全卡住。现无Token对照证明mean pooling不足，本轮不做大编码器微调。若继续研究新结构，先固定训练组、开发准则和全新封存录制，再做小型Token-readout对照。','','未训练或控制π₀.₅/OpenPI/RTC，未连接机器人，未推送远程。','']
    Path('docs/TOMATO_REFRESH101_RESULTS_20261109.md').write_text('\n'.join(lines));print('REPORT COMPLETE; gate',output['gate_passed'],'CI',output['paired_recall_delta_ci95'])

if __name__=='__main__':run()
