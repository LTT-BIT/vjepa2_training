"""Validate30Hz video lengths, exact source-frame CSV alignment and frozen labels."""
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from . import tomato_dataset as ds
from .tomato_decision_attribution import ROOT
from .validate_tomato_boundary_reviews import validate


def run():
    root=ROOT/'review_package';data=ds.read_json(root/'cases.json');cases=data['cases'];names=data['joint_names']
    assert len(cases)==41 and len({c['case_id'] for c in cases})==41
    assert sum('strict_miss' in c['tags'] for c in cases)==16
    assert sum('all_adjust_pickup' in c['tags'] for c in cases)==9
    assert sum('early_prompt_1p3s' in c['tags'] for c in cases)==1
    assert sum('unmatched_warning' in c['tags'] for c in cases)==23
    def check(c):
        p=root/c['video'];r=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=avg_frame_rate,nb_read_frames,start_time','-of','json',str(p)],text=True))['streams'][0]
        expected=c['clip_end_frame']-c['clip_start_frame'];assert int(r['nb_read_frames'])==expected
        assert r['avg_frame_rate']=='30/1' and float(r['start_time'])==0
        with (root/c['state_csv']).open() as f:rows=list(csv.reader(f))
        assert rows[0][2:60]==names
        actual_frames=np.array([int(row[0]) for row in rows[1:]])
        assert np.array_equal(actual_frames,np.arange(c['clip_start_frame'],c['clip_end_frame'])),c['case_id']
        assert np.all(np.diff([float(row[1]) for row in rows[1:]])>0)
        for key in ['position_velocity_plot','probability_plot','trace_json']:assert (root/c[key]).exists()
        assert ds.digest(c['annotation_path'])==c['annotation_sha256']
        return dict(case_id=c['case_id'],frames=expected,fps=r['avg_frame_rate'],video_csv_source_frames_exact=True,annotation_unchanged=True)
    rows=list(ThreadPoolExecutor(max_workers=4).map(check,cases))
    valid=validate(ds.read_json(root/'review_opinions_template.json'),data)
    assert all(v['proposed_boundary_frame'] is None for v in ds.read_json(root/'review_opinions_template.json')['reviews'])
    result=dict(passed=True,cases=rows,opinion_validation=valid,ground_truth_modified=False)
    ds.write_json(ROOT/'package_checks.json',result);print('41 review cases: video/frameCSV/annotation hashes/opinion identities PASS')


if __name__=='__main__':run()
