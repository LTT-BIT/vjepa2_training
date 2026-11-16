"""CLI for frozen-prediction replay; GT only supplied to scorer after each replay ends."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from .tomato_prompt_replay import read,write,sha,jsonl,preflight,replay
from .evaluate_tomato_prompt_switch import score_episode,aggregate,calibration

ROOT=Path(__file__).resolve().parents[1]


def csv_write(path, rows):
    if not rows:Path(path).write_text('');return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def run_case(manifests,streams,integrity,model,split,policy,out,latency=.09,period=.5,offset=0.,lock_sha=None,compressed=False,only_episode_ids=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    config=dict(policy);rows=[];failures=[];all_events=[];all_intervals=[];replays={};calibrations=[]
    selected_episodes=[e for e in manifests[model]['episodes'] if e['split']==split and
                       (only_episode_ids is None or e['episode_id'] in only_episode_ids)]
    for e in selected_episodes:
        replays[e['episode_id']]=replay(streams[model][e['episode_id']],config,e['length']/e['fps'],latency,period,offset)
    for e in selected_episodes:
        eid=e['episode_id'];r=replays[eid]
        # All policy episodes have ended before GT enters the independent scorer.
        metrics=score_episode(e,r);rows.append(metrics)
        if any(c['time']+1e-9<c['request']['source_available_at'] or c['time']+1e-9<c['request']['release_at'] for c in r['commits']):
            raise AssertionError('Noncausal commit')
        if len({c['time'] for c in r['commits']})!=len(r['commits']):raise AssertionError('Multiple commits on one boundary')
        calibrations.append(calibration(streams[model][eid],e))
        common=dict(episode_id=eid,source_recording_id=e['source_recording_id'],model=model,split=split)
        all_events.extend({**common,**v} for v in r['events'])
        all_intervals.extend({**common,**v} for v in r['intervals'])
        for t in metrics['transitions']:
            if not t['matched'] or t['commit_delta_s']<=-.25+1e-9:
                failures.append(dict(**common,type='missed_transition' if not t['matched'] else 'premature_commit',
                    severity=3 if not t['matched'] else 4,time=t['time'],**{k:v for k,v in t.items() if k!='time'}))
        matched_times={t['commit_time'] for t in metrics['transitions'] if t['matched']}
        for c in r['commits']:
            if c['time'] not in matched_times:failures.append(dict(**common,type='unmatched_commit',severity=2,time=c['time'],commit=c))
        if metrics['wrong_prompt_duration_s']>2:
            failures.append(dict(**common,type='episode_wrong_duration',severity=1,time=0.,wrong_duration_s=metrics['wrong_prompt_duration_s']))
    report=aggregate(rows);report.update(model=model,split=split,policy=config,latency_s=latency,plan_period_s=period,
        plan_offset_s=offset,selected_policy_lock_sha256=lock_sha,
        matching_definition='Max cardinality monotone one-to-one exact stage-pair match in [-1,+2] seconds; min absolute delta then lexicographic indices',
        on_time_definition='Matched delta in [-0.2,+0.5] divided by all eligible boundaries; strict matching separately reported',
        wrong_to_stage_definition='New prompt differs from GT at commit time, includes correctly anticipated but early stages',
        early_wrong_definition='Wrong active equals the immediate next GT segment during current GT interval',
        event_metrics_are_not_raw_forecast_metrics=True)
    n=sum(c['samples'] for c in calibrations)
    report['raw_forecast_calibration']=dict(samples=n,nll=sum(c['nll']*c['samples'] for c in calibrations)/n,
        brier=sum(c['brier']*c['samples'] for c in calibrations)/n,
        ece_10_equal_width=sum(abs(sum((c['bins'][i]['mean_confidence']-c['bins'][i]['accuracy'])*c['bins'][i]['count'] for c in calibrations if c['bins'][i]['count'])) for i in range(10))/n,
        diagnostic_only=True,calibrated=False)
    report['raw_forecast_calibration']['per_true_class']={}
    for label in ['pickup','harvest','return','adjust']:
        pp=[c['per_true_class'][label] for c in calibrations];nn=sum(v['samples'] for v in pp)
        report['raw_forecast_calibration']['per_true_class'][label]=dict(samples=nn,
            nll=sum(v['nll']*v['samples'] for v in pp if v['samples'])/nn if nn else None,
            brier=sum(v['brier']*v['samples'] for v in pp if v['samples'])/nn if nn else None,
            ece_10_equal_width=sum(abs(sum(v['confidence_bins'][i]['confidence_sum']-v['confidence_bins'][i]['correct_sum'] for v in pp)) for i in range(10))/nn if nn else None,
            reason=None if nn else 'no_class_targets')
    for name,rr in [('replay_events',all_events),('active_prompt_intervals',all_intervals)]:
        if compressed:
            with (out/(name+'.jsonl.gz')).open('wb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
                    for row in rr:f.write((json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n').encode())
        else:jsonl(out/(name+'.jsonl'),rr)
    write(out/'evaluation_control.json',report)
    csv_write(out/'per_episode_metrics.csv',[{k:v for k,v in r.items() if k not in ['timing','transitions']} for r in rows])
    csv_write(out/'per_transition_metrics.csv',[dict(episode_id=r['episode_id'],source_recording_id=r['source_recording_id'],**t) for r in rows for t in r['transitions']])
    jsonl(out/'failure_cases.jsonl',sorted(failures,key=lambda r:(-r['severity'],r['time'],r['source_recording_id'])))
    write(out/'replay_integrity.json',dict(success_sources=len(rows),failed_sources=0,input_anchors=report['input_anchors'],consumed_anchors=report['consumed_anchors'],alignment_errors=[],input_sha256=integrity['input_sha256']))
    write(out/'replay_config.lock.json',dict(policy=config,latency_s=latency,plan_period_s=period,plan_offset_s=offset,
        model=model,split=split,selected_policy_lock_sha256=lock_sha,event_priority='prediction(0), timer/release(1), planning(2); stop after all t<end',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').glob('*tomato_prompt*.py')},
        input_sha256=integrity['input_sha256']))
    return dict(summary=report,per_episode=rows,replays=replays)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,default=ROOT/'runs/tomato/local93_20261107')
    p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--model',choices=['future_only','joint','current_only'],default='future_only')
    p.add_argument('--policy-config',type=Path,default=ROOT/'configs/tomato_prompt_switch_offline.yaml')
    p.add_argument('--policy-id',default='P0_raw');p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261108);p.add_argument('--latency-s',type=float,default=.09)
    p.add_argument('--plan-period-s',type=float,default=.5);p.add_argument('--plan-offset-s',type=float,default=0.)
    p.add_argument('--selected-policy-lock',type=Path)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    config=read(a.policy_config)
    policy=next(v for v in config['policies'] if v['name']==a.policy_id)
    policy={**policy,'prediction_ttl_s':config['prediction_ttl_s'],'initial_prompt':config['initial_prompt'],'seed':a.seed}
    if a.split=='test' and not a.selected_policy_lock:raise ValueError('Test requires a validation-selected policy lock')
    lock_sha=None
    if a.selected_policy_lock:
        lock=read(a.selected_policy_lock);lock_sha=sha(a.selected_policy_lock)
        if policy!=lock['policy'] or sha(a.policy_config)!=lock['policy_config_sha256']:raise ValueError('Policy lock mismatch')
    m,s,i=preflight(a.run_root)
    run_case(m,s,i,a.model,a.split,policy,a.output_dir,a.latency_s,a.plan_period_s,a.plan_offset_s,lock_sha)


if __name__=='__main__':main()
