"""Read-only named state compatibility audit; does not run or modify OpenPI ROI."""
import argparse
import ast
from pathlib import Path
import subprocess
import numpy as np
import pyarrow.parquet as pq
from . import tomato_dataset as ds
from .tomato_durable_memory import select_named_positions
from .tomato_frozen_protocol import verify


def roi_position_columns(path):
    tree=ast.parse(Path(path).read_text());found=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=='q' for t in node.targets):continue
        v=node.value
        if not isinstance(v,ast.Subscript) or not isinstance(v.value,ast.Attribute) or v.value.attr!='r_':continue
        if not isinstance(v.slice,ast.Tuple):continue
        columns=[]
        for sub in v.slice.elts:
            if not isinstance(sub,ast.Subscript) or not isinstance(sub.value,ast.Name) or sub.value.id!='state' or not isinstance(sub.slice,ast.Slice):raise ValueError('Unrecognized ROI state expression')
            start=0 if sub.slice.lower is None else ast.literal_eval(sub.slice.lower);end=ast.literal_eval(sub.slice.upper)
            if sub.slice.step is not None or not isinstance(start,int) or not isinstance(end,int):raise ValueError('Unrecognized ROI state slice')
            columns.extend(range(start,end))
        found.append(columns)
    if len(found)!=1 or len(found[0])!=58:raise ValueError('Require one explicit ROI 58D named-layout audit target')
    return found[0]


def run(out):
    if out.exists():raise FileExistsError('Preserve previous audit')
    lock=verify(Path('configs/tomato_frozen_independent_20261112/b0.lock.v3.json'));m=ds.read_json(lock['manifest']);target=m['contract']['joint_names'];by={r['episode_id']:r for r in ds.read_json('runs/tomato/complete100_20261111/data_identity.json')['episodes']};roi=Path('/home/user/tomato-openpi/scripts/tomato_roi_locator.py');columns=roi_position_columns(roi);rows=[];layout=None
    for e in m['episodes']:
        if e['split']=='test':continue
        names=e['info']['features']['observation.state']['names'];table=pq.read_table(e['parquet'],columns=['observation.state']);raw=np.asarray(table['observation.state'].to_pylist(),np.float32);mapped=select_named_positions(raw,names,target);times,existing=ds.read_state(e,target,m['contract']['joint_mode']);assert np.array_equal(mapped,existing)
        path=Path(by[e['episode_id']]['feature_path']);assert ds.digest(path)==by[e['episode_id']]['feature_sha256'];z=np.load(path);expected=mapped[z['input_indices']].reshape(len(z['anchors']),928);assert np.array_equal(expected,z['features'][:,-928:])
        roi_names=[names[i] for i in columns];converted=select_named_positions(raw[:,columns],roi_names,target);assert np.array_equal(converted,mapped)
        permutation=[roi_names.index(n) for n in target]
        if layout is None:layout=dict(source_dimension=len(names),roi_position_names=roi_names,agent_position_names=target,roi_to_agent_permutation=permutation,orders_identical=roi_names==target)
        else:assert layout['roi_position_names']==roi_names and layout['source_dimension']==len(names)
        rows.append(dict(episode_id=e['episode_id'],split=e['split'],source_recording_id=e['source_recording_id'],source_schema_sha256=ds.fingerprint(names),cache_sha256=ds.digest(path),cached_windows=len(z['anchors']),named_mapping_exact=True,all_cached_history_positions_exact=True))
    out.mkdir(parents=True);ds.write_json(out/'state_mapping.json',dict(episodes=rows,layout=layout,historical_test_scored=False,roi_code_sha256=ds.digest(roi),roi_code_path=str(roi),openpi_git_sha=subprocess.check_output(['git','-C','/home/user/tomato-openpi','rev-parse','HEAD'],text=True).strip(),roi_executed=False,openpi_modified=False,full100_four_view_cache_ready=False));print('NAMED_STATE_VERIFIED',len(rows),'orders_identical',layout['orders_identical'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
