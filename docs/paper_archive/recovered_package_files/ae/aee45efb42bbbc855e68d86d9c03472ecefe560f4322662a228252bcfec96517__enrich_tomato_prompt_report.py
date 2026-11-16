"""Post-hoc report enrichment and independent saved-event audit; no policy changes."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import zipfile
from .tomato_prompt_replay import read,sha,write
from .simulate_tomato_prompt_switch import ROOT
from .report_tomato_prompt_switch import table,f


def load_lines(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as handle:
        for line in handle:yield json.loads(line)


def enrich(out):
    out=Path(out);lock=read(out/'selected_policy.lock.json')
    for p,h in lock['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Locked implementation changed: '+p)
    root=ROOT/'runs/tomato/local93_20261107'
    manifest=read(root/'joint/manifest.json');episodes={e['episode_id']:e for e in manifest['episodes']}
    audit=dict(cases=0,episodes=0,commits=0,predictions=0,consumed_unscorable_tail_predictions=0,
        interval_coverage_errors=0,duration_recompute_errors=0,clock_errors=0,whitelist_errors=0,locked_sources_unchanged=True)
    for summary_path in sorted(out.rglob('evaluation_control.json')):
        case=summary_path.parent;summary=read(summary_path)
        af=case/'active_prompt_intervals.jsonl';ef=case/'replay_events.jsonl'
        if not af.exists():af=af.with_suffix('.jsonl.gz')
        if not ef.exists():ef=ef.with_suffix('.jsonl.gz')
        intervals={}
        for row in load_lines(af):intervals.setdefault(row['episode_id'],[]).append(row)
        csv_rows={r['episode_id']:r for r in csv.DictReader((case/'per_episode_metrics.csv').open())}
        for eid,aa in intervals.items():
            ep=episodes[eid];duration=ep['length']/ep['fps'];gt=ep['annotation']['segments']
            if abs(aa[0]['start_s'])>1e-9 or abs(aa[-1]['end_s']-duration)>1e-9:audit['interval_coverage_errors']+=1
            if any(abs(a['end_s']-b['start_s'])>1e-9 for a,b in zip(aa,aa[1:])):audit['interval_coverage_errors']+=1
            # Independent midpoint sweep over every GT/Active breakpoint, not the scorer's nested overlap implementation.
            points=sorted(set([0.,duration]+[v['start_s'] for v in aa]+[v['end_s'] for v in aa]+
                              [s['start_frame']/ep['fps'] for s in gt]))
            wrong=0.
            for left,right in zip(points,points[1:]):
                t=(left+right)/2
                actual=next(v['stage'] for v in aa if v['start_s']<=t<v['end_s'])
                truth=next(s['label'] for s in gt if s['start_frame']/ep['fps']<=t<s['end_frame']/ep['fps'])
                if actual!=truth:wrong+=right-left
            if abs(wrong-float(csv_rows[eid]['wrong_prompt_duration_s']))>1e-7:audit['duration_recompute_errors']+=1
            audit['episodes']+=1
        last={};commit_times=set()
        for ev in load_lines(ef):
            eid=ev['episode_id'];t=ev['time']
            if t<last.get(eid,-1)-1e-9:audit['clock_errors']+=1
            last[eid]=t
            if ev['kind']=='prediction':
                msg=ev['input'];audit['predictions']+=1
                allowed={'episode_id','source_recording_id','anchor_frame','observed_at','available_at','model_name',
                         'future_probabilities','joint_current_probabilities','horizon_s'}
                if set(msg)!=allowed or abs(t-msg['available_at'])>1e-9 or abs(ev['source_target_time']-msg['observed_at']-msg['horizon_s'])>1e-9:
                    audit['whitelist_errors']+=1
                if msg['horizon_s']>0 and msg['observed_at']+msg['horizon_s']>=episodes[eid]['length']/episodes[eid]['fps']:
                    audit['consumed_unscorable_tail_predictions']+=1
            elif ev['kind']=='commit':
                audit['commits']+=1;p=ev['request'];key=(eid,t)
                phase=(t-summary['plan_offset_s'])/summary['plan_period_s']
                if (t<p['source_available_at']-1e-9 or t<p['release_at']-1e-9 or abs(phase-round(phase))>1e-7
                    or key in commit_times or abs(p['source_target_time']-p['source_observed_at']-(0 if summary['model']=='current_only' else 1))>1e-9):
                    audit['clock_errors']+=1
                commit_times.add(key)
        audit['cases']+=1
    for key in ['interval_coverage_errors','duration_recompute_errors','clock_errors','whitelist_errors']:
        if audit[key]:raise AssertionError((key,audit[key]))
    protected=read(out/'protected_original_sha256.json')
    if any(sha(p)!=h for p,h in protected.items()):raise AssertionError('Original result changed')
    write(out/'independent_artifact_audit.json',audit)
    raw={m:read(root/m/'evaluation.json')['splits']['test']['future'] for m in ['future_only','joint']}
    for m in raw:
        current=read(out/'test_main'/m/'evaluation_control.json')['raw_forecast_calibration']
        assert abs(raw[m]['nll']-current['nll'])<1e-12
    write(out/'raw_forecast_reference.json',{m:{k:r[k] for k in ['macro_f1','accuracy','nll','anticipation']} for m,r in raw.items()})
    matrix=read(out/'sensitivity_matrix.json');selected=lock['name']
    aux=read(out/'auxiliary_comparison.json')
    ranges=[]
    for m in ['future_only','joint']:
        rr=[v for v in matrix if v['split']=='test' and v['candidate']==selected and v['model']==m]
        ranges.append([m,f"{min(r['matched_transitions'] for r in rr)}–{max(r['matched_transitions'] for r in rr)} /54",
            f"{min(r['active_prompt_time_accuracy'] for r in rr):.2%}–{max(r['active_prompt_time_accuracy'] for r in rr):.2%}",
            f"{min(r['commit_delta_s']['p50'] for r in rr):.3f}–{max(r['commit_delta_s']['p50'] for r in rr):.3f}"])
    addition=['## 13. 完成后的独立复算与结果解读（不改变策略）',
        '本节在全部预注册实验结束后仅核验保存的事件和补充解读，不重放新策略、不修改参数或模型。',
        '### 实际回答研究问题',
        '1. **没有全面胜出的模型。** 主测试Joint Future匹配48/54，高于Future-only的45/54，提前≥0.5秒也更少（1 vs 2）；但时长一致率略低（94.75% vs 94.98%），完整顺序8/9低于9/9，匹配提交中位数更迟（1.033 vs 0.867秒）。配对来源CI跨0，不能声称稳定控制优势。',
        '2. **置信度/合法边能改变行为，但当前风险与覆盖尚未同时达标。** val的P1_t90相较P0改善完整顺序（B:4→6，C:2→7），却仍未令两模型共同满足预声明提前风险限制。P4将两者顺序都约束到9/9，但B时长一致率降至93.99%，并仍有8个提前≥0.5秒的匹配。顺序更整齐不等于切换时序更安全。',
        '3. **本次Current硬门控不可用。** 在已锁定defer=0.5秒的规则下，该门控在val/test都没有commit，转换匹配0/54，Active长期停留pickup；test错误Prompt达1133.567秒。等待期间Current已改变，门控撤回Pending；Active一旦滞后，又因Current不一致而无法恢复。这是该具体“与Active不一致就冻结”规则的实测失败，不能推广为所有Current辅助无价值。未根据此test结果实现替代门控。',
        '4. **反应式基线与未来策略也有取舍。** 同后端下Current-only test匹配47/54，略高于Future-only45/54，但时长一致率93.11%较低、Δt中位数1.633秒较迟、完整顺序8/9较低；其提前≥0.5秒为0。不能仅用单一匹配数选择部署模型。',
        '5. **主要损失已经从提前风险转为确认/规划迟滞。** 共享选中的P3策略需两个2Hz锚点，至少增加0.5秒确认，再defer 0.5秒，并等规划边界；1秒预测先行量可能被这些等待耗尽。严格[-0.2,+0.5]窗口仅B:9/54、C:5/54。没有证据支持“未来分类好，就可以现在安全切换”。',
        '### 与旧分类指标严格分栏',
        table(['原始冻结forecast（非控制）','Test Macro-F1','原始边界前事件','NLL'],[[m,f(raw[m]['macro_f1']),f"{raw[m]['anticipation']['detected_events']}/{raw[m]['anticipation']['evaluable_events']}",f(raw[m]['nll'])] for m in raw]),
        '上述分类/原始预警值来自原报告，仅作为参照；主报告45/54和48/54是后端actual commit的有序匹配，定义不同。',
        '### P6范围解读（所有场景预先约定，并未选择测试最优配置）',
        table(['锁定策略/test','转换匹配范围','时长一致率范围','匹配Δt中位数范围秒'],ranges),
        '主要失效模式是规划网格错过释放时刻而增加整段等待、确认/释放期间候选撤回、Active偏离后难以回到合法顺序，以及门控锁死。模型置信度变化和时钟相位共同影响状态路径；不能把矩阵最优相位当作已验证部署参数。',
        '### 独立保存产物审计',
        f"复核{audit['cases']}个真实case、{audit['episodes']}条Episode回放、{audit['predictions']}个可用输入事件、{audit['commits']}次commit。对保存的Active/GT所有断点做独立中点积分，错误时长与CSV一致；所有区间连续覆盖，commit均不早于支持证据/释放时刻，且位于规划边界、每边界最多一次。错误数均为0。",
        f"实际消费{audit['consumed_unscorable_tail_predictions']}条未来监督目标已超出录像支持的尾部消息，证明回放没有用future_valid提前裁去尾部。策略消息严格白名单、原始target不随HOLD移动。原321个历史文件和锁定源码哈希保持不变。",
        f"复算JSON：`{out/'independent_artifact_audit.json'}`；原始分类参照：`{out/'raw_forecast_reference.json'}`。复现补充步骤：`.venv-tomato/bin/python -m tools.enrich_tomato_prompt_report {out}`。新增该报告后处理工具，不改变锁中的任何策略、评分或仿真源码。"]
    report=ROOT/'docs/TOMATO_LOCAL93_PROMPT_SWITCH_SIM_REPORT.md'
    if '## 13.' in report.read_text():raise FileExistsError('Report already enriched')
    report.write_text(report.read_text()+'\n\n'+'\n\n'.join(addition)+'\n')
    (out/'TOMATO_LOCAL93_PROMPT_SWITCH_SIM_REPORT.md').write_text(report.read_text())
    hashes=read(out/'artifact_sha256.json')
    for p in [report,out/'TOMATO_LOCAL93_PROMPT_SWITCH_SIM_REPORT.md',out/'independent_artifact_audit.json',out/'raw_forecast_reference.json',Path(__file__).resolve()]:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    write(out/'artifact_sha256.json',hashes)
    sources=[p for p in (ROOT/'tools').glob('*tomato_prompt*.py')]+list((ROOT/'tests').glob('test_tomato_prompt*.py'))+[ROOT/'configs/tomato_prompt_switch_offline.yaml']
    # Repackage report additions; immutable selected policy and numeric experiment outputs stay untouched.
    with zipfile.ZipFile(out/'Tomato_Prompt_Switch_Evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        files=[p for p in out.rglob('*') if p.is_file() and p.suffix!='.zip' and not
            ('sensitivity' in p.relative_to(out).parts and p.name.endswith('.gz'))]
        for p in files+sources+[report]:z.write(p,str(p.relative_to(ROOT)))
    print(audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output_dir',type=Path);enrich(parser.parse_args().output_dir)
