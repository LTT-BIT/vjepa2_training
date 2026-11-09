"""Versioned evidence report, read-only audit and portable human-review archive."""
import collections
import shutil
import subprocess
import zipfile
from pathlib import Path
from . import tomato_dataset as ds
from .tomato_decision_attribution import ROOT,SOURCE
from .audit_tomato_transitions import manifest


def run():
    dest=Path('docs/validation/tomato_boundary_review_20261109');dest.mkdir(exist_ok=True)
    summary=ds.read_json(ROOT/'decision_summary.json');events=ds.read_json(ROOT/'event_attribution.json')['events'];old=ds.read_json(SOURCE/'events.json')['events'];oldby={e['event_id']:e for e in old}
    traces=ds.read_json(ROOT/'early_prompt_trace.json');decomp=ds.read_json(ROOT/'prompt_timing_decomposition.json');data=ds.read_json(ROOT/'review_package/cases.json');package=ROOT/'review_package';lock=ds.read_json(ROOT/'protocol.json')
    expectations={}
    for path in ['runs/tomato/optimization_20261108/freeze.json','runs/tomato/temporal_20261108_v2/frozen_prior.json','runs/tomato/long_history_20261108_v2/frozen_source_baseline.json']:
        x=ds.read_json(path);expectations.update(x.get('prior_artifacts',x))
    expectations.update(lock['read_only_sources']);expectations.update(ds.read_json(ROOT/'unchanged_model_configs.json'))
    changed=[p for p,sha in expectations.items() if not Path(p).exists() or ds.digest(p)!=sha]
    m,eps=manifest();ann_changed=[]
    for e in m['episodes']:
        actual=ds.digest(e['annotation']['path'])
        if actual!=e['annotation']['sha256']:
            current=ds.read_json(e['annotation']['path'])
            ann_changed.append(dict(episode_id=e['episode_id'],split=e['split'],source_recording_id=e['source_recording_id'],path=e['annotation']['path'],expected_sha256=e['annotation']['sha256'],actual_sha256=actual,frozen_segments=e['annotation']['segments'],current_segments=current['segments'],current_revision=current.get('revision'),mtime=Path(e['annotation']['path']).stat().st_mtime,agent_modified=False))
    assert not changed and not any(e['split']=='val' for e in ann_changed)
    active_now=dict(branch=subprocess.check_output(['git','-C','/home/user/tomato-openpi','branch','--show-current'],text=True).strip(),sha=subprocess.check_output(['git','-C','/home/user/tomato-openpi','rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','-C','/home/user/tomato-openpi','status','--short'],text=True).strip())
    integrity=dict(passed=not ann_changed,validation_evidence_passed=True,global_annotation_snapshot_intact=not ann_changed,historical_files_checked=len(expectations),changed=changed,annotations_checked=len(m['episodes']),changed_annotations=ann_changed,
        training_runs=0,encoder_calls=0,test_inference_calls=0,source_split_sha256=m['split_sha256'],historical_source=lock['historical_openpi'],active_openpi_start=lock['live_openpi'],active_openpi_at_report=active_now,
        statement='Historical bridge isolated from33ce417; active OpenPI branch left untouched. Model candidate and stable default configs unchanged.')
    ds.write_json(ROOT/'integrity.json',integrity)
    ds.write_json(ROOT/'training_annotation_drift.json',dict(items=ann_changed,validation_annotations_unchanged=True,training_performed=False,source_file_restored_or_modified=False,action='Keep frozen checkpoint/cache/val evidence. Reconcile data version before any future retraining; do not mix current source labels with old cache targets.'))
    for name in ['decision_summary.json','event_attribution.json','early_prompt_trace.json','prompt_timing_decomposition.json','package_checks.json','integrity.json','training_annotation_drift.json']:
        ds.write_json(dest/name,ds.read_json(ROOT/name))
    # Preliminary observations are not human adjudications or suggested ground truth.
    notes=[]
    for case in data['cases']:
        e=next(x for x in events if x['event_id']==case['target_event_id'])
        note=dict(case_id=case['case_id'],target_event_id=case['target_event_id'],source_annotation_sha256=case['annotation_sha256'],original_boundary_frame=case['original_boundary_frame'],human_review_status='pending',analyst_type='automated_initial_triage',
            proposed_boundary_frame=None,earliest_plausible_frame=None,latest_plausible_frame=None,
            suggested_search_window_frames=[case['clip_start_frame'],case['clip_end_frame']-1],search_window_is_not_label_uncertainty=True,
            model_evidence=dict(future_target_samples_in_valid_window=e['window_target_argmax_count'],future_strict_k1_hit=e['future_strict_k1']['detected'],future_strict_k2_hit=e['future_strict_k2']['detected']),
            question='复核原阶段开始/结束动作锚点、遮挡、重复/失败尝试；不能根据模型高概率自动移动边界',revision_history=[])
        if case['from_stage']=='adjust' and case['to_stage']=='pickup':note['question']='区分盘内重定位与新的拿起抓握序列；分别标接触、闭合、带动物体、离盘。不要把握住不抬起预先当作负例。'
        if case['case_id']=='3e645bf4e137d0f836c9d7ff_b4':
            note['agent_visual_triage']='已查看70、71、71.5、72、73.3、74秒三视角帧：71.5/72秒物体仍靠近盘，围拢/调整的接触语义无法仅凭静帧确定。没有证据据模型把标签前移1.3秒，建议保留原标签至两人复核。'
            note['visual_evidence_path']=str((package/case['case_id']/'semantic_review_contact.jpg').resolve())
        notes.append(note)
    ds.write_json(ROOT/'initial_triage_notes.json',dict(human_opinions_completed=0,notes=notes,labels_revised=0,policy='No fabricated human consensus; proposed boundaries remain null until human review.'))
    ds.write_json(dest/'initial_triage_notes.json',ds.read_json(ROOT/'initial_triage_notes.json'))
    ds.write_json(dest/'review_case_index.json',dict(convention=data['convention'],cases=[{k:v for k,v in c.items() if k not in ['current_probabilities','future_probabilities','current_future_times_s']} for c in data['cases']]))
    lines=['# 番茄 Agent：阶段边界复核与决策误差分解（2026-11-09）','',
      '核实起点：research-tomato-agent / 412fbc17861f19b7187e7c78236c476bf43316b9，工作区干净。没有启动训练、编码器前向或测试集推断，原76/10/8和模型均保持。发现1条训练原始标注与历史清单不同，已单独记录；本轮使用的10条验证标注完全匹配冻结版本。固定概率来自既有独立1.5秒双头seed42、2Hz验证输出。',
      '', '## 主要结论','',
      '40/56的主要限制是Future预测的时间定位及部分类别路径错误，不是单纯K次确认或七阶段约束。16次漏检中，8次有效预警窗内没有足够的目标类别证据，5次目标类别过早进入稳定状态、有效窗内不再发生新转换，3次短预警被K2确认延后或打断。标签是否在个别案例中有歧义尚待人工复核，不能直接把偏差解释为标注错误。',
      '', 'adjust→pickup的9次中，3次过早、4次没有有效窗内目标证据，2次成功；K1和K2都只命中2/9。改变确认等待无法修复这7次错位。训练中该转移只有76次，而pickup→harvest及harvest→return各152次；验证该类9次，其他两类各19次。数量和动作条件都值得补充，但现有数据未系统标遮挡/速度/停顿，不能据次数直接断言因果。',
      '', '继续使用独立1.5秒双头、2Hz研究候选，保留Joint seed42和原接口默认配置。新增联合决策仅研究对照，没有改生产接口。没有找到值得替换当前候选的可靠性提升。',
      '', '## 复核包：41个项目','',
      '16次漏检与全部9次adjust→pickup合并后为18个真实边界，其中包含提前Prompt案例；另有23个未匹配预警，共41个项目。每项有保留30Hz的三路视频、Current/Future原始2Hz概率、原边界/模型0.5-K2边界、58D命名关节位置及真实时间戳差分速度、完整状态CSV和实际Prompt/Chunk追踪。模型概率曲线之间连线不等于新增观测。',
      '', f'[打开复核页面](http://127.0.0.1:8768/)；本地入口：`{(package/"index.html").resolve()}`。可按漏检、误报、全部adjust→pickup、提前Prompt筛选；逐帧步进、0.25倍速、切换58个关节，导出带源SHA和修订历史的意见JSON。默认隐藏模型/原边界，显示后记录model_seen。原标注从不写回。',
      '', '该包经过模型错误筛选，不能称为独立盲测；隐藏网页内容不构成盲法。两位复核者须用独立浏览器档案/机器分别导出，第三方保存共识。当前人工意见完成数为0：已有自动初筛意见、待填建议帧和区间，未伪造人工裁决。检索范围只是浏览范围，不是已经确认的标注不确定区间。',
      '', '标注规范见docs/TOMATO_BOUNDARY_ANNOTATION_STANDARD_V1.md：pickup以新的拿起抓握序列开始为锚点，不能等成功离盘；harvest以可见摘取接近开始；return以持物返程开始；adjust以盘支撑后的重新定位开始。失败尝试/真实回退保留，七个位置不强制补齐。规范为需试标确认的v1审议稿，本轮不据此替换历史标签。',
      '', '## 指标和因果边界','',
      '所有方法使用同56个真实转换和原一对一精确from/to匹配：Current允许[b,min(b+2秒,新阶段结束))；Future允许[b−1.5秒,b)。首次稳定类别只是初始化，不计事件。未匹配预测均算误报。Current决策模型不接收人标边界；只有明确标记的oracle诊断使用真值，不可部署，也不是模型成绩。',
      '', '分类：Current Macro-F1=0.974811；Future1.5 Macro-F1=0.923555。事件：严格Future40/56。实际接口按下一个有序阶段发level warning的宽松覆盖47/56，三种定义分开。边界不确定性不纳入本轮改分，仅保留复核记录。',
      '', '## Current决策固定消融','',
      '|方法|事件命中/56|误报数|误报/min|命中事件中位延迟s|', '|---|---:|---:|---:|---:|']
    for name,r in summary['current'].items():lines.append(f'|{name}|{r["hits"]}|{r["false_events"]}|{r["false_per_minute"]:.3f}|{r["conditional_offset_p50_s"]:.3f}|')
    lines += ['', '置信度比较固定0.5/0.8，确认固定K1/K2，单因素诊断后不选择新阈值。argmax到门控、门控到确认的命中变化同时受事件时刻和错误路径影响，不可简单相减当作互不重叠的因果贡献。完整逐事件变化在prompt_timing_decomposition.json。',
      '', '0.8/K2无序四类与有序七阶段都命中51/56；有序约束只将误报从6减为5，本验证集没有由七阶段约束额外造成的Current事件漏检。不能外推到回退/缺失阶段场景。',
      '', '联合规则是：Current和Future当前都以≥0.8预测同一个期待的下一阶段，连续2次才能确认；Future不能单独推进阶段。输入严格因果。结果49/56、误报7次，劣于原有序Current的51/56、误报5次；6d46…_b2/b3受损。该规则把“当前阶段”与“1.5秒后阶段”简单要求相同，存在语义局限，不能据此否定所有联合策略。它也没有阻止提前Prompt案例，因为两头当时都相信pickup。',
      '', '## Future确认与模型误差','', '|方法|严格命中/56|误报数|误报/min|命中事件中位提前s|', '|---|---:|---:|---:|---:|']
    for name,r in summary['future'].items():lines.append(f'|{name}|{r["hits"]}|{r["false_events"]}|{r["false_per_minute"]:.3f}|{-r["conditional_offset_p50_s"]:.3f}|')
    lines += ['', 'K1/.5为38/56、50次误报，K2/.5为40/56、23次误报。K1救回3个被确认等待损失的事件，却失去K2下另外5个事件，净下降2。因而“全部16个漏检来自K2过慢”被反驳。0.8/K2虽少1个误报，但也少2个命中，不据此更换阈值。',
      '', f'相同采样与K2下，oracle Current、Future分别命中{summary["oracles"]["current"]["hits"]}/56、{summary["oracles"]["future"]["hits"]}/56，无误报。说明该验证集本身没有不可避免的采样/确认事件上限40；oracle使用真值仅用于离线诊断，不能看作实际方案。',
      '', '## 全部adjust→pickup逐例证据','', '|事件|标注秒|有效窗目标样本数|K1/K2命中|最近K2同路径事件偏移s|诊断|', '|---|---:|---:|---|---|---|']
    for e in events:
        if e['from_stage']!='adjust' or e['to_stage']!='pickup':continue
        orig=oldby[e['event_id']];dt=orig.get('nearest_same_transition_offset_s');hit=e['future_strict_k2']['detected'];diagnosis='命中' if hit else '过早且持续为目标类，无窗内新事件' if e['window_target_argmax_count'] else '有效窗没有足够目标证据；最近同路径转换迟到'
        # Earlier report only saves nearest offset on misses; matched offset from strict result.
        if hit:dt=e['future_strict_k2']['offset_s']
        lines.append(f'|{e["event_id"]}|{e["boundary_s"]:.3f}|{e["window_target_argmax_count"]}|{e["future_strict_k1"]["detected"]}/{hit}|{dt:.3f}|{diagnosis}|')
    lines += ['', '3个过早案例仍可在有效窗内看到稳定pickup概率，不能把它们改称严格事件成功：新类别进入时间本来就早于预测时距允许范围，是时距定位错误。原有七阶段level warning可利用持续概率，但那是另一层的覆盖指标，不能拿来替代40/56。4个迟到案例在有效窗内没有达到正确目标argmax/.5，不是再多等一次就能提前预测。',
      '', '## 提前1.3秒Prompt的完整链路','',
      '完整Episode历史先回放，再截取68.3—76.3秒附近日志。原始标注为frame2199/73.3秒；71.5秒(frame2145) Current pickup≈0.9923，FSM仍在adjust、Prompt version3，计数0→1；72.0秒(frame2160) pickup≈0.9889，计数达到2，FSM转pickup_2、Prompt version3→4、计数重置0。旧version3 Chunk匹配失败，新version4 Chunk匹配成功。',
      '', '直接触发来源是Current证据提前，接口依照已登记规则执行；确认等待增加了0.5秒延迟，反而把相对标注的−1.8秒推迟到−1.3秒。移除Future预警不改变Prompt，说明不是Future预警越权更新。对所有56个最终确认事件都验证了：Prompt偏移 = 当前成功连续证据串起点相对边界偏移 + 0.5秒确认等待。最终全部56个位置都确认，但严格时窗内仅51个；提前1个、过晚4个。',
      '', '已实际查看70、71、71.5、72、73.3、74秒三路帧。71.5/72秒物体仍靠近盘，无法仅据静态握姿分清盘内调整与新拿起抓握；需人工标闭合、带动物体、离盘三个锚点。没有证据直接将人工边界前移1.3秒，更不能用模型置信度定义真值。本轮结论是“相对冻结标注的Current提前识别＋正常确认行为”，而非已证实原标注错或接口缺陷。',
      '', '新增单元测试复现了两个高置信Current预测触发提前转换，并验证Future单独预警不会更新Prompt、断续确认会重置、旧Chunk失效、单向阶段约束及因果前缀一致性。没有发现明确接口契约缺陷，所以未修改OpenPI或RTC；并未把提前行为强行改写为测试失败再调阈值。',
      '', '## 问题、可执行改进与验证','',
      '|问题层|证据|下一步操作|验证方式|','|---|---|---|---|',
      '|模型分类/预测|Current整体F1高，但局部阶段起点提前；Future存在过早、迟到和错误路径|用复核锚点分清阶段语义；补重复调整、抓握未拿起、裸梗长等待的实录和条件标签|冻结模型在新开发录制逐事件检验，分别报告4类和不同转换；不按测试结果调参|',
      '|决策确认|K2有3次损失、5次收益；0.5秒等待可精确分解|保留现行K2；后续若提议可变确认，单因素预注册并检查提前Prompt和误报|同一概率轨迹、同一56事件配对比较，再在新开发组复核|',
      '|阶段约束|本验证集没有额外漏检，只少1次误报；异常回退覆盖不足|保留七阶段接口；把非标准序列单列人工复核/拒绝策略研究|用缺失阶段、回退和中止的新录制检验，不伪造七阶段标签|',
      '|联合决策|简单同意门控损失2个Current事件，不能消除提前Prompt|不采用该启发式；若研究一致性，应使用时距语义而非假定两头同一时刻标签应相同|先固定因果规则做离线反事实，再独立开发录制消融|',
      '|人工不确定性|adjust与新抓握外观相近，尚无两位独立复核者共识|41项人工复核、辅助锚点、最早/最晚区间、原始/建议点和修订历史|统计人际差异和区间，不只重标错例；统一新版本后才另行评价|',
      '|Prompt效果|1次提前1.3秒、4次晚于允许窗；Future不能直接切Prompt|保留原逻辑和候选，先核对语义与实际Current起点|真实历史接口逐观测日志、版本/Chunk检验；不连接硬件|',
      '', '## 新发现：训练原始标注版本漂移','',
      '全量核查发现训练Episode e80436dea962fa58288fd8f9（原录制651eb2271bae4971a62273985ed85c2d）当前原始标注revision2与冻结manifest不同：最后harvest→return从frame2989（99.633秒）移到2908（96.933秒），末段return从1帧变为82帧。新旧SHA及区间保存在training_annotation_drift.json。本轮未写该文件，不能从本轮日志确定修改来源，也没有恢复/覆盖它。',
      '', '这说明原始数据树不再整体等同历史94条快照，不能宣称94份标注全部未变。10条验证标注均匹配，56事件和已有模型得分不受影响；1197份历史产物也一致。未来重训前应确认该修订的来源/批准记录，建立新manifest和目标标签/缓存身份；旧训练数据版本及已有结果仍冻结。这个harvest→return样本的问题不能用来解释全部adjust→pickup失败。',
      '', '## 补录与真正盲测','',
      '具体清单在docs/TOMATO_RECORDING_CHECKLIST_20261109.md：12条开发、20条主封存测试，另4条预登记备用；6个开发场次组、10个主测试场次组、2个备用组。补足不同速度、重复调整、抓握不抬起、遮挡、裸梗停顿、返程/释放重叠及缺失/回退。不能预设困难动作必然是负例，应按真实动作标注。计划数据尚未采集。',
      '', '以原录制/植株/场次分组，开发与盲测不共享；尽量跨新日期和操作者，若有重叠如实报告。锁定checkpoint、归一化、Schema、规范、2Hz/1.5秒、阈值和事件口径，再释放封存测试一次验收。备用只按事件数量缺口或技术失败启用，不按分数补数据。新的盲测一旦用于修改模型/阈值即视为已暴露。',
      '', '## 代码、检查与适用限制','',
      '开始核查时OpenPI主目录处于tomato-subtasks-ab-train / 7b83057，末次核对该分支已推进到6242e4a，工作区干净；检出中没有agent_bridge.py。本轮从其Git对象33ce417只读提取原三份接口/契约源码，保存SHA到本轮隔离目录，并复现上轮有序回放。没有切换、修改或恢复OpenPI工作目录。因此这里验收的是历史固定接口，不是声称当前AB训练分支已经具备同一接口。',
      '', f'134项回归/单元测试通过；41个视频的30Hz、零起点、逐帧数量，以及状态CSV的源帧号、命名关节、原标注SHA逐项通过。历史{integrity["historical_files_checked"]}份已冻结产物及候选/默认配置SHA未变；94份原始标注中93份匹配、1份训练标注漂移，10条验证均匹配。训练0次、编码器前向0次、测试推断0次。',
      '', '人工复核工具检查中修复两项本轮新建辅助功能问题：浮点秒数裁剪会让14个片段CSV首尾最多偏1帧，已改用源帧索引并重新生成/逐项验证；普通静态HTTP服务不支持视频Range导致步进失败，已提供仅loopback的Range服务器并浏览器实测1帧步进68.000→68.033秒。它们不是原始训练数据或OpenPI缺陷。',
      '', f'完整本地产物：`{ROOT.resolve()}`。人审包包含原始实测曲线/CSV和视频，不进入Git；版本化代码、规范、逐事件诊断和结果摘要。意见仍需人工填写，本轮没有自行完成双人专家重标。',
      '', '复现命令（在仓库根目录）：','', '```bash',
      '.venv-tomato/bin/python -m tools.tomato_decision_attribution',
      '.venv-tomato/bin/python -m tools.decompose_tomato_prompt_timing',
      'PYTHONPATH=/tmp/tomato-transition-plotdeps .venv-tomato/bin/python -m tools.build_tomato_boundary_review',
      '.venv-tomato/bin/python -m tools.check_tomato_review_package',
      '.venv-tomato/bin/python -m tools.plan_tomato_blind_recordings',
      '.venv-tomato/bin/python -m tools.report_tomato_boundary_review',
      '.venv-tomato/bin/python -m tools.serve_tomato_boundary_review --port 8768',
      '```','', '历史接口快照由本轮源SHA锁定；在新机器须从33ce417提取后验证，不得静默导入当前其他分支接口。绘图复用上轮隔离依赖，不修改训练环境。']
    Path('docs/TOMATO_BOUNDARY_REVIEW_AND_DECISIONS_20261109.md').write_text('\n'.join(lines)+'\n')
    # Portable reviewer instructions and a stdlib-only server.
    shutil.copyfile('docs/TOMATO_BOUNDARY_ANNOTATION_STANDARD_V1.md',package/'ANNOTATION_STANDARD.md')
    shutil.copyfile('tools/serve_tomato_boundary_review.py',package/'run_review_server.py')
    readme=['# 41案例人工复核包','', '这是历史验证数据的模型辅助复核包，不是新的独立盲测。建议两位复核者在不同机器/浏览器档案分别操作。不要共用浏览器意见缓存。','',
        '在本目录启动：`python3 run_review_server.py --directory . --port 8768`，打开 http://127.0.0.1:8768/ 。服务器仅绑定本机，支持逐帧视频定位所需的HTTP Range。直接双击HTML可能无法读取CSV。','',
        '先看视频和位置/速度，记录可观察锚点，再显示原边界和模型。逐帧步进以30Hz源帧号表示；概率只来自2Hz，不把曲线插值当成新观测。切换前保存意见并导出JSON；导出只包含已保存意见。网页不会写回原标注。','',
        '建议帧与合理区间可为空，不能用高模型概率替代人工动作证据。给出修订建议时须填复核者、理由和区间。查看模型后model_seen保持true；重新隐藏不会恢复未暴露状态。','',
        '导出意见在研究仓库用tools.validate_tomato_boundary_reviews校验结构；最终标注需另行两人复核和共识，本包没有自动修改或批准标签的功能。']
    (package/'README.md').write_text('\n'.join(readme)+'\n')
    # Source diagnostics remain local; generated review fields stay empty for real humans.
    ds.write_json(ROOT/'review_artifact_manifest.json',dict(files={str(p.relative_to(package)):ds.digest(p) for p in package.rglob('*') if p.is_file()},source_protocol_sha256=ds.digest(ROOT/'protocol.json')))
    archive=ROOT/'tomato_boundary_review_41cases.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in package.rglob('*'):
            if p.is_file():z.write(p,arcname=str(Path('tomato_boundary_review')/p.relative_to(package)))
    print('REPORT, FROZEN-INTEGRITY CHECKS AND PORTABLE41-CASE ARCHIVE COMPLETE')


if __name__=='__main__':run()
