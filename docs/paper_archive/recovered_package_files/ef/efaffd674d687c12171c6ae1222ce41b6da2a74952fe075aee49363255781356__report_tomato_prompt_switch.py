"""Final evidence report; no policy selection or test-driven parameter changes."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from .tomato_prompt_replay import read,write,sha,jsonl
from .simulate_tomato_prompt_switch import ROOT


def table(head,rr):
    esc=lambda v:str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(esc,head))+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(esc,row))+' |' for row in rr])
def f(v):return f'{v:.3f}' if v is not None else 'null'
def metric_row(name,r):
    return [name,f"{r['matched_transitions']}/{r['eligible_transitions']}",f"{r['active_prompt_time_accuracy']:.2%}",
            f(r['wrong_prompt_duration_s']),r['wrong_to_stage_commits'],r['unmatched_commits'],r['premature_ge_05'],
            f"{r['negative_delta_count']}/{r['zero_delta_count']}/{r['positive_delta_count']}",
            f"{f(r['commit_delta_s']['p50'])}/{f(r['commit_delta_s']['p95'])}",
            f"{r['full_episode_order_exact_count']}/{r['episodes']}"]


def report(out):
    out=Path(out);lock=read(out/'selected_policy.lock.json');config=read(out/'selection_protocol.json');selected=lock['name']
    test={model:read(out/'test_main'/model/'evaluation_control.json') for model in ['future_only','joint']}
    val={model:read(out/'val_grid'/selected/model/'evaluation_control.json') for model in test}
    ci=read(out/'test_cluster_bootstrap.json');matrix=read(out/'sensitivity_matrix.json');aux=read(out/'auxiliary_comparison.json')
    integrity=read(out/'replay_integrity.json');status=read(out/'stage_status.json');completion=read(out/'completion.json')
    with (out/'regression_tests.log').open('w') as log:
        command=[sys.executable,'-m','pytest','-q',*[str(p) for p in sorted((ROOT/'tests').glob('test_*tomato*.py'))],str(ROOT/'tests/test_lemon_vjepa_phase.py')]
        proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    status['FINAL_regression']='PASS' if proc.returncode==0 else 'FAIL';write(out/'stage_status.json',status)
    rows=list(csv.DictReader((out/'val_candidates.csv').open()));paired=list(csv.DictReader((out/'test_paired_sources.csv').open()))
    special=read(out/'special_three_stage_summary.json');visual=read(out/'plots/visualization_integrity.json')
    b=test['future_only'];c=test['joint']
    header=['模型/集合','实际commit匹配','时长一致率','错误Prompt秒','commit时与GT不符','未匹配commit','提前≥0.5秒','提前/同时/滞后','Δt P50/P95秒','完整顺序正确']
    lines=['# LOCAL93：冻结预测驱动的离线Prompt切换仿真报告',
        '## 1. 摘要与最终建议',
        f"实际HEAD `{completion['git_head']}`；共享策略仅用9条val来源锁定为 **{selected}**，参数 `{json.dumps(lock['policy'],ensure_ascii=False)}`。主场景假设预测到达延迟90ms，规划周期0.5秒、相位0。风险约束是否满足：**{lock['constraints_satisfied']}**。如果为false，该策略只是预声明fallback排序的最优项，不代表风险已达标。",
        table(header,[metric_row('Future-only / test',b),metric_row('Joint Future / test',c)]),
        '建议仅继续离线/Shadow Mode研究，尚不足以采用“未来Softmax高置信度即执行切换”的真机路线。未来标签预测不是动作完成检测；GT时间一致率、事件匹配与完整顺序分别描述不同现象。以下结果不证明机器人安全，也不能从测试结果回头选择新策略。',
        '## 2. 实验输入与可信性',
        f"预检93/93来源通过：75 train、9 val、9 test，三模式每个27,293个2Hz锚点，顺序 `pickup/harvest/return/adjust`。序列分布 `{integrity['sequence_counts']}`。没有排除来源或补预测。主对照实际执行9 val+9 test来源；额外三阶段资料在其原集合单独检查，不加入主测试。",
        '配对键为(source_recording_id,episode_id,frame_index,timestamp_s)，允许1e-6秒验证容差；拒绝重复、缺失、倒序、未来输入索引、概率非法和split混淆。manifest、split、任务契约及来源/标注快照逐模式比较；reviewed标注文件SHA验证。checkpoint仅做SHA锁定，不加载或更改。完整输入SHA见 replay_integrity.json；原始输出保护见 protected_original_sha256.json。',
        f"三模式共保护 `{read(out/'original_artifact_integrity.json')['protected_files']}` 个历史输出文件，结束时全部SHA不变。冻结特征、原视频/Parquet/标注未写入。已有用户未提交文件保留；本轮新增模块独立于旧训练/评估代码。",
        table(['状态项','实际状态'],status.items()),
        '## 3. P1 因果回放审计',
        '策略只接收frozen/slots PolicyInput白名单，不含任何GT、future_valid、future_target_time_s或录制长度。Forecast target由original observed_at+1s计算，Current-only反应式输入为observed_at+0s。状态由固定启动先验pickup开始，全程不按真实边界或错误重置。',
        '事件时间四舍五入到1ns；同一时刻先接收结果，再处理expiry/release，最后规划边界提交，结束事件最后。每个确认必须来自不同新锚点；planning HOLD不能增加计数。TTL=0.75秒，从最后一条新结果可用时间计，严格超时才失效；Active继续保持，Pending取消，不借TTL自动切换。新证据撤回或候选变化会取消Pending并重置计数。',
        '回放驱动器用Episode长度安排输入停止，但该长度从未传给策略。只处理t<数据终止的可用消息；无法在终止前完成的尾部结果不消费，其余尾部预测即使future_valid=false也照常处理。数据停止只产生stream_end，不产生FINISH。每个case全部Episode回放完成后才调用独立GT评分。',
        f"真实selected Joint val复跑：`{read(out/'determinism.json')}`；完整事件、Active区间、评估JSON逐字节相同。全流程主进程torch_imported={completion['torch_imported']}，transformers_imported={completion['transformers_imported']}。未启动Encoder/CUDA。",
        '## 4. P2 策略与参数表',
        table(['name','策略','θ','K','margin','defer秒'],[[v[k] for k in ['name','id','theta','K','margin','d']] for v in config['policies']]),
        'P0直接argmax；P1增加置信度；P2增加新锚点连续确认/概率margin；P3增加Pending释放延迟；P4增加合法边pickup→harvest→return→adjust→pickup。P2_k1与P1_t80、P3_d0与P2_k2故意保留为实现等价性检查，不是更多独立搜索。未尝试全笛卡尔积、模型特定调参、TTL/时距扫描或温度校准。',
        '## 5. P3 独立控制级指标',
        'Active区间与GT半开区间精确求交得错误时长；时长一致率=1−错误秒/总秒。early_wrong是错误Active等于紧邻下一GT阶段的重叠时长，其余单列。错误commit定义为commit时新Active与GT不一致，因此包括相对GT提前的正确下一阶段，不等同于真实物理错误。extra_commits=max(0,commit数−GT边界数)；unmatched_commits按匹配结果计算。',
        '边界匹配按每个Episode、精确阶段对、单调一对一动态规划：主窗口[-1,+2]秒内先最大化匹配数，再最小化|Δt|总和，最后按配对索引字典序破同分。首个pickup不算边界。独立严格窗口[-0.2,+0.5]重新匹配；on_time_ratio是主匹配中落入该窗口的数量/全部边界。提前≥0.25/0.5/1秒仅为人工边界相对风险诊断，不是任务提前完成。',
        table(['test模型','strict匹配','on_time','提前≥0.25/0.5/1秒','Δt最小/最大秒','early wrong/other wrong秒','非法候选/撤销/过期'],[
            [model,f"{r['strict_window_matched']}/{r['eligible_transitions']}",f"{r['on_time_count']}/{r['eligible_transitions']}",
             f"{r['premature_ge_025']}/{r['premature_ge_05']}/{r['premature_ge_1']}",f"{f(r['commit_delta_s']['min'])}/{f(r['commit_delta_s']['max'])}",
             f"{f(r['early_wrong_prompt_duration_s'])}/{f(r['other_wrong_prompt_duration_s'])}",f"{r['illegal_edge_attempts']}/{r['canceled_pending']}/{r['stale_prediction_count']}"] for model,r in test.items()]),
        '首次错误commit后恢复时间按实际Active与GT再次一致的最早时间事后计算，未人工复位；没有错误或未恢复分别用null+reason。完整per_episode_metrics.csv包含每类时长、错误时长、序列、恢复状态。短阶段/1秒内多边界不从时长评分删除；事件有无歧义敏感性和return首次/最终标记保留在JSON/CSV。',
        '## 6. P4 公平对照与验证锁定',
        'selection_protocol.json在任何新test控制评分前保存。两个模型分别满足错误时长≤5%、提前≥0.5秒的匹配数≤10%边界、事件召回≥80%，才视为可行；可行项按B/C等权平均召回最大、平均错误时长比例最小、平均匹配|Δt|最小、名称排序。若均不可行，按两模型归一化约束超出总量、错误时长、提前比例、负召回及名称排序。阈值为预声明研究约束，不是安全标准。',
        table(['val候选','模型','匹配','时长一致率','提前≥0.5秒','完整顺序'],[[r['candidate'],r['model'],f"{r['matched_transitions']}/{r['eligible_transitions']}",f"{float(r['active_prompt_time_accuracy']):.2%}",r['premature_ge_05'],f"{r['full_episode_order_exact_count']}/{r['episodes']}"] for r in rows]),
        f"共享锁文件SHA `{completion['selection_lock_sha256']}`。主test及P6输出均绑定此SHA；辅助试验的固定规则也在锁中预声明。未依据test更改阈值、算法、checkpoint、时距或规划周期。",
        table(header,[metric_row('B / val',val['future_only']),metric_row('C / val',val['joint'])]),
        table(['test source','B/C匹配','B/C时长一致率','C−B百分点','B/C完整顺序'],[[r['source_recording_id'],f"{r['B_matches']}/{r['C_matches']} of {r['eligible']}",f"{float(r['B_time_accuracy']):.2%}/{float(r['C_time_accuracy']):.2%}",f"{float(r['C_minus_B_time_accuracy'])*100:+.3f}",f"{r['B_exact']}/{r['C_exact']}"] for r in paired]),
        '95%配对来源cluster bootstrap（5000次，seed20261108），方向均为C−B：',
        table(['指标','95%百分位CI'],[[k,f'[{f(v[0])}, {f(v[1])}]'] for k,v in ci['intervals'].items()]),
        f"来源时长一致率胜/负/平 `{ci['source_wins_losses_ties_time_accuracy']}`。只有9独立test来源，CI仅探索性；错误秒/提前计数CI为同样9个重采样来源总量，不能解读为每Episode值。",
        '### 独立附加对照',
        table(header,[metric_row(k,r) for k,r in aux.items()]),
        'Joint Current辅助规则是：Current argmax与Active不同时冻结该条Future请求并撤回Pending，除此与共享策略相同。它可能抑制误切换，也可能在Active落后时阻止恢复；只作附加比较，不混入B/C公平主对照。Current-only使用相同提交骨架，输入目标是当前阶段，不把它冒充未来预测。',
        '### Softmax概率诊断（未拟合校准）',
        table(['模型/test','有效未来目标','NLL','Brier(四类平方和)','ECE(10等宽bin)'],[[k,r['raw_forecast_calibration']['samples'],*[f(r['raw_forecast_calibration'][v]) for v in ['nll','brier','ece_10_equal_width']]] for k,r in test.items()]),
        '每真实类别的NLL/Brier/ECE见evaluation_control.json。这些指标仅评价forecast target分类概率，不是可以安全切换概率。只有独立评分器在未来目标超出资料时跳过概率诊断，策略仍消费可用的尾部预测。',
        '## 7. P5 自动可视化与异常索引',
        f"自动生成{visual['episodes']}个val/test Episode ×3类图，共{visual['png_count']}张PNG：完整时轴、四类概率、每个边界±3秒细节。自动PNG解码检查通过。P5_manual_review=**AUTO_GENERATED; MANUAL_REVIEW_DEFERRED / NOT_PERFORMED**。没有打开、解码、重编码MP4或逐帧核验，没有改标注或据图调参。",
        f"图与异常索引：`{out/'plots/timeline_index.md'}`。每个case含failure_cases.jsonl，按风险、时间、来源排序；原始observed_at/available_at、固定target及actual commit时间分别保存。图上raw future位于原始observation时间，目标始终是该anchor+1秒；HOLD不伪造新预测。",
        '## 8. P6 假设延迟与规划时钟敏感性',
        f"真实执行{len(matrix)}个case：val/test各9来源 ×延迟[0,.09,.2,.3] ×周期[.25,.5,1] ×相位[0,周期/2] ×共享锁定策略及P0/P2参考 ×两模型。每case的输入/消费锚点数、实际Episode数、匹配、错误/提前、Pending、时延分解均见sensitivity_matrix.json/.csv。若参考与锁定策略同名会去重，真实case数以文件为准。",
        '延迟是注入的结果可用时间假设，数据到达延迟假设为0；没有测量真实模型计算/传输时间。所有延迟<0.5秒锚点间隔，采用单服务串行无拥塞队列，overload/drop_late/jitter为NOT_RUN，不能外推超载吞吐。π0.5的15步规划仅由参数化时钟代替，未测每步时间，也未实现RTC/T3动作块。',
        table(['test:latency/period/offset秒','B匹配','C匹配','B/C一致率','B/C提前≥0.5秒','B/C Δt P50秒'],[
            [f"{r['latency_s']}/{r['plan_period_s']}/{r['plan_offset_s']}",f"{r['matched_transitions']}/{r['eligible_transitions']}",
             f"{cc['matched_transitions']}/{cc['eligible_transitions']}",f"{r['active_prompt_time_accuracy']:.2%}/{cc['active_prompt_time_accuracy']:.2%}",
             f"{r['premature_ge_05']}/{cc['premature_ge_05']}",f"{f(r['commit_delta_s']['p50'])}/{f(cc['commit_delta_s']['p50'])}"]
            for r in matrix if r['split']=='test' and r['candidate']==selected and r['model']=='future_only'
            for cc in [next(c for c in matrix if c['split']==r['split'] and c['candidate']==r['candidate'] and c['model']=='joint' and (c['latency_s'],c['plan_period_s'],c['plan_offset_s'])==(r['latency_s'],r['plan_period_s'],r['plan_offset_s']))]]),
        '每次commit均自动检查不早于其证据available_at与release_at，且每规划边界最多一次。时延分解：首候选观测→可用、K确认等待、defer等待、规划等待，四者相加等于首候选观测→commit；最新支持锚点的预测年龄另算。置信度、取消和闭环Active状态会随相位改变，因此不同场景的匹配集合/整体指标不保证单调；不将不同证据的两个commit当成同一事件作时间倒流判断。孤立相同消息的0/90/200/300ms因果时钟另有单元测试。',
        '## 9. 困难案例、特殊Episode与风险',
        table(['特殊三阶段模型','来源ID','原始集合','期望序列','实际Active序列','匹配','FINISH'],[[k,v['per_episode'][0]['source_recording_id'],integrity['sequences'][v['per_episode'][0]['episode_id']]['split'],v['per_episode'][0]['gt_sequence'],v['per_episode'][0]['active_sequence'],f"{v['summary']['matched_transitions']}/{v['summary']['eligible_transitions']}",v['per_episode'][0]['finish_status']] for k,v in special.items()]),
        '四类合法边只约束局部阶段对，不能区分第一/第二轮return，也无法确定最终return之后应停止；特殊三阶段按自然序列评分，没有伪装七阶段。first/final return和短阶段标记已保留。数据结束≠FINISH已检测。',
        '风险重点是相对标注过早harvest→return/ pickup→harvest、非法跳转尝试、K确认造成漏检或迟滞、Pending证据撤销、合法边拒绝导致错误状态长期保持。完整failures和每类时长供后续独立复核，本轮不对物理抓取完成或放回安全作判断。',
        '## 10. 与既有93条训练报告的衔接',
        '旧报告Future-only原始未来事件50/54、Joint Future 48/54，只要求边界前1秒至少一次预测下一阶段；本报告统计已经经过门槛、K、defer、规划时钟而真正提交的Active。不能把旧50/54或48/54当控制器检出，也不能将test Future Macro-F1较高直接解释为切换更好。原分类训练报告、checkpoint、预测日志没有修改。',
        '## 11. 下一步与证据边界',
        '不推荐据本轮test再挑策略或规划相位。当前四类概率可继续用于离线或Shadow提示研究，但是否需要Transition Head/Completion Head应由新的独立来源与任务完成定义支持；本轮既不实现新头，也不能证明完整HFSM必需或可省略。优先补充独立保留来源、真实5090+π0.5并发可用时延/队列测量、动作块更新协议和人工物理完成核验，再作上线决策。',
        '未执行：人工视频复核、重新标注、真实GPU并发推理测速、π0.5/ROS/RTC/T3/HFSM集成、终止检测、新数据采集、过载队列及jitter。所有0/90/200/300ms结果只能称为假设实时参数下的离线开放环轨迹回放；Prompt变更没有生成新的机器人动作轨迹。',
        '## 12. 复现、测试与附件清单',
        f"仿真/自动图表阶段用时{completion['duration_s']:.1f}秒（不含最终全回归与打包）。Python `{completion['python'].splitlines()[0]}`。核心回放/策略/评分仅标准库；绘图依赖matplotlib/Pillow，未导入PyTorch/transformers。",
        '```bash\ncd /home/user/pi05/vjepa2_training\n.venv-tomato/bin/python -m compare_unused_placeholder\n```'.replace('.venv-tomato/bin/python -m compare_unused_placeholder',f'.venv-tomato/bin/python -u -m tools.compare_tomato_prompt_policies --output-dir {out}'),
        '该命令是本次实际入口；输出目录已存在会拒绝覆盖。若要重现请换新目录，协议不变。单case CLI：',
        f'```bash\n.venv-tomato/bin/python -m tools.simulate_tomato_prompt_switch --run-root {ROOT}/runs/tomato/local93_20261107 --split test --model future_only --policy-config configs/tomato_prompt_switch_offline.yaml --policy-id {selected} --selected-policy-lock {out}/selected_policy.lock.json --output-dir /tmp/tomato_prompt_reproduce_new --latency-s 0.09 --plan-period-s 0.5 --seed 20261108\n```',
        '完整测试实际输出：\n\n```text\n'+(out/'regression_tests.log').read_text().strip()+'\n```',
        '新增文件：6个实现模块（policy/replay/scorer/CLI/comparison/visualize）及本报告生成器、独立offline配置、3个测试模块；仅新增，不修改旧模块。与建议目录相比，将latency/policy边界测试合并在test_tomato_prompt_replay.py，评估与比较各独立模块，避免重复fixture。',
        '实施记录：首轮开发运行在val网格结束后、生成selected_policy.lock及任何test控制评分之前，因CSV导出重复model关键字失败。修复序列化并增加端到端编排回归；首轮产物保留，最终在新目录重跑。阈值、候选网格和选择排序未改动；没有依据test修补策略。取消事件另明确保存canceled旧请求与pending=null的变化后状态，决策行为不变。',
        f"完整产物根目录：`{out}`。val_grid含所有12候选B/C；test_main为锁定主对照；auxiliary为额外信息对照；special_three_stage为特殊资料；sensitivity为各时钟场景。每case提供evaluation_control.json、逐Episode/逐转换CSV、failure_cases.jsonl、replay_config.lock.json及完整events/Active区间；P6日志使用确定性gzip压缩(.jsonl.gz)节省空间，内容仍可独立重算。",
        'selection_protocol.json、selected_policy.lock.json、val_selection_rankings.json、test_paired_sources.csv、test_cluster_bootstrap.json、sensitivity_matrix.csv/json、plots/timeline_index.md及artifact_sha256.json提供完整追踪。精简证据ZIP包含主对照、全部val网格汇总/指标、场景矩阵、图表和源码测试；完整P6事件留在本地并由SHA清单引用。没有自动commit或push。']
    path=ROOT/'docs/TOMATO_LOCAL93_PROMPT_SWITCH_SIM_REPORT.md'
    if path.exists():raise FileExistsError('Final report already exists; do not overwrite historical report')
    path.write_text('\n\n'.join(lines)+'\n');(out/'TOMATO_LOCAL93_PROMPT_SWITCH_SIM_REPORT.md').write_text(path.read_text())
    sources=[p for p in (ROOT/'tools').glob('*tomato_prompt*.py')]+list((ROOT/'tests').glob('test_tomato_prompt*.py'))+[ROOT/'configs/tomato_prompt_switch_offline.yaml']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.zip'}
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in sources+[path]})
    write(out/'artifact_sha256.json',hashes)
    with zipfile.ZipFile(out/'Tomato_Prompt_Switch_Evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        files=[p for p in out.rglob('*') if p.is_file() and p.suffix!='.zip' and not
            ('sensitivity' in p.relative_to(out).parts and p.name.endswith('.gz'))]
        for p in files+sources+[path]:z.write(p,str(p.relative_to(ROOT)))
    if proc.returncode:raise RuntimeError('Regression gate FAIL; inspect report')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output_dir',type=Path);report(p.parse_args().output_dir)
