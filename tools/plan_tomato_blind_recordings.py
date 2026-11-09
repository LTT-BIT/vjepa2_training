"""Concrete prospective recording manifest; no recordings are claimed collected."""
from pathlib import Path
from . import tomato_dataset as ds
from .tomato_decision_attribution import ROOT


def run():
    dev=[
      ('adjust_pickup','slow','adjust后静止5-8秒，再闭合、带动物体、离盘分别留可见证据'),
      ('adjust_pickup','fast','连续调整后快速再次拿起，禁止为了对齐模型刻意停顿'),
      ('adjust_pickup','normal','重复盘内调整3次，只接近/闭合后撤2次，最后才真正拿起'),
      ('adjust_pickup','normal','抓握后握持但不抬起5秒；标明确pickup尝试或继续adjust，不能预设负例'),
      ('adjust_pickup','normal','手掌遮挡局部接触，至少一路相机能看到物体与盘的分离'),
      ('adjust_pickup','slow','手腕视角部分离画，远景保留物体和盘，标缺失观测而不补帧'),
      ('return_adjust','normal','返程、触盘、释放、盘内重新定位四个锚点分开'),
      ('return_adjust','fast','触盘与手指调整连贯重叠，保留合理边界区间'),
      ('harvest_return','slow','剩裸梗后继续等待8秒，随后才返程，覆盖提前8.1秒误报类型'),
      ('pickup_harvest','normal','拿起持稳后左手失败尝试/重复摘取，记录阶段内子事件'),
      ('full_sequence','normal','新果串形态、照明、取放速度组合；完整七阶段'),
      ('noncanonical','normal','只前三阶段/缺失adjust/任务中止，不补齐七阶段')]
    test_conditions=[
      ('adjust_pickup','slow','调整完成后随机长度停顿，区分闭合与离盘'),
      ('adjust_pickup','fast','快速连续调整再拿起'),
      ('adjust_pickup','normal','接近后撤、重复调整、握住不拿起等困难条件；按可见动作区分真实pickup尝试与负例'),
      ('adjust_pickup','normal','局部手指遮挡，三路中至少一路可见关键物体运动'),
      ('harvest_return','slow','裸梗/末果状态长停顿，正常返程对照'),
      ('return_adjust','normal','放下/松手/重定位重叠与明确分离各一条'),
      ('pickup_harvest','normal','左手接近/失败摘取/持稳等待，阶段内多次尝试'),
      ('full_sequence','normal','新的物体形态与光照，完整七阶段'),
      ('partial_sequence','normal','三阶段终止或缺失adjust，按实际序列评估'),
      ('noncanonical','normal','返程中止/回退/异常重复，标真实序列并单报适配器失配')]
    rows=[]
    for i,(focus,speed,condition) in enumerate(dev,1):
        group=f'DEV_G{(i+1)//2:02d}';rows.append(dict(recording_id=f'DEV_{i:02d}',role='development',group_id=group,plant_id=group+'_PLANT',session_day=f'DEV_DAY{(i-1)//4+1}',operator_role='DEV_OPERATOR',focus=focus,speed=speed,condition=condition,annotation='双人独立锚点标注+区间；先不看模型',collected=False))
    for g,(focus,speed,condition) in enumerate(test_conditions,1):
        for repetition in [1,2]:
            group=f'BLIND_G{g:02d}';rows.append(dict(recording_id=f'BLIND_{(g-1)*2+repetition:02d}',role='sealed_independent_test',group_id=group,plant_id=group+'_PLANT',session_day=f'BLIND_DAY{(g-1)//2+1}',operator_role='UNSEEN_TEST_OPERATOR_IF_AVAILABLE',focus=focus,speed=speed,condition=condition+f'；组内独立录制{repetition}',annotation='不含模型文件的真实盲标包；锁定后一次评价',collected=False))
    for i in range(1,5):
        group=f'BLIND_G{10+(i+1)//2:02d}'
        rows.append(dict(recording_id=f'BLIND_RESERVE_{i:02d}',role='sealed_test_reserve',group_id=group,plant_id=group+'_PLANT',session_day=f'BLIND_RESERVE_DAY{(i+1)//2}',operator_role='UNSEEN_TEST_OPERATOR_IF_AVAILABLE',focus='adjust_pickup',speed='slow' if i%2 else 'normal',condition='仅由保管者按事件数量缺口或技术失败启用；新植株完整七阶段；绝不依据模型分数追加',annotation='同主盲测包，先锁定后释放',collected=False))
    cfg=ds.read_json('configs/tomato_transition_20261109/recommended_inference.json')
    plan=dict(status='planned_not_collected',development=12,blind=20,reserve=4,blind_groups=10,reserve_groups=2,recordings=rows,locked_candidate_model_version=cfg['model_version'],reference='Joint seed42 unchanged',
      acceptance=dict(strict_future_recall_min=.8,false_events_per_minute_max=1,current_event_recall_min=.9,current_p95_delay_s_max=1.5,premature_prompt_count_report_separately=True),
      sealing='Before release, lock model SHA, source/schema/norm SHA, annotation convention, thresholds .5/K2 analysis and .8/K2 Current provider, horizon1.5 and2Hz, one-to-one matching, exclusions and statistical analysis. Collector/custodian retains new test videos/labels; developers cannot access before lock. No model-containing HTML used for true blind test.',
      sample_size='Aim>=100 real blind transitions, with>=20 adjust->pickup if naturally available in complete sequences; current20-row plan includes4 partial/noncanonical clips, so exact counts may be lower. Add prespecified whole NEW blind groups based on missing event counts/technical failures only, never on model scores.',
      group_policy='No shared raw recording/crop/reencoding/plant/session between development and blind. Prefer different operators and days; if unavailable report overlap rather than claim unknown-operator generalization. Group uncertainty by10 session/plant groups, not video windows.',
      noncanonical='Primary canonical seven-stage events reported separately from incomplete/backtracking sequences. Never force labels into seven phases. Report abstention/review need and premature prompts for out-of-contract sequences.',
      exposure='After one locked primary evaluation, any subsequent tuning makes that test exposed; a new independent test is then needed. Do not pick a different threshold/model based on highest test score.')
    ds.write_json(ROOT/'recording_plan.json',plan)
    dest=Path('docs/validation/tomato_boundary_review_20261109');dest.mkdir(exist_ok=True);ds.write_json(dest/'recording_plan.json',plan)
    text=['# 新录制逐条清单与盲测方案','', '计划12条开发、20条主封存测试，另列4条条件启用的备用测试；共6个开发场次组、10个主测试场次组、2个备用场次组。尚未采集。全部使用真实三相机/实测58D位置与原时间戳，保留接触、闭合、带动物体、离盘、返程、触盘、释放等人工辅助锚点。', '', '|ID|用途/分组|速度|重点|具体录制要求|','|---|---|---|---|---|']
    for r in rows:text.append(f'|{r["recording_id"]}|{r["role"]}/{r["group_id"]}|{r["speed"]}|{r["focus"]}|{r["condition"]}|')
    text+=['','## 锁定与保管','',plan['sealing'],'',plan['sample_size'],'',plan['group_policy'],'',plan['noncanonical'],'',plan['exposure'],'',
      '主要缺口证据：训练转换pickup→harvest/harvest→return各152次，而return→adjust/adjust→pickup各76次；验证后两类各9次。仅凭次数不能确认遮挡/停顿/重复动作的覆盖，故必须补条件标签。adjust→pickup7次失败全部可由过早3次/迟到4次描述，优先补这些条件，而非复制稳定harvest长片段。', '',
      '真正独立测试不能使用含模型曲线但CSS隐藏的复核网页。先发无模型输出的原始标注包；两位盲标者各自留存意见，再形成共识。开发者先锁定候选、Joint参考及所有规则，保管者再释放测试数据给锁定脚本一次评价；保存完整身份清单与首次访问记录。', '',
      '规范v1先在开发录制试标。若与历史规范的动作语义存在系统差别，须预先声明评价版本，不能在看到测试分数后选标注定义。保留原标注、辅助锚点、分歧区间和修订历史。20条录制并非20个完全独立场景；按10个场次/植株组报告不确定性和每类支持数。']
    Path('docs/TOMATO_RECORDING_CHECKLIST_20261109.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':run()
