"""Lazy CPU head-only inference; never extracts features or trains."""
from pathlib import Path
from .catalog import read

def infer(catalog,eid,mode):
    import torch
    torch.set_num_threads(4)
    from .. import tomato_phase_model as model,tomato_metrics as metrics
    from ..train_tomato_vjepa_phase import prediction_rows
    from .catalog import sha
    e=catalog.get(eid);m=catalog.manifests[mode];run=catalog.root/mode
    index=read(run/'feature_index.json')
    if index['manifest_sha256']!=sha(run/'manifest.json'):raise ValueError('Feature index manifest mismatch')
    item=next((r for r in index['episodes'] if r['episode_id']==eid),None)
    if item is None:raise FileNotFoundError('该样本没有可用特征，请先按原工具提取')
    path=Path(item['path'])
    if not path.exists():raise FileNotFoundError('该样本没有可用特征，请先按原工具提取')
    model.assert_source_unchanged(e)
    d=model.load_cache(path,e,m['contract'],index['pretrained'])
    saved,head=model.load_checkpoint(run/'best_head.pt',sha(run/'manifest.json'),'cpu',m['split_sha256'])
    if saved['prediction_mode']!=mode:raise ValueError('Checkpoint mode mismatch')
    logits=model.predict_heads(head,d['features'],saved['mean'],saved['std'])
    probs={h:metrics.probabilities(v) if v is not None else None for h,v in [('current',logits['current_logits']),('future',logits['future_logits'])]}
    entry=dict(episode=e,**d)
    rows=prediction_rows(entry,probs,e['split'],m['prediction'],{h:logits[h+'_logits'] for h in ['current','future']})
    return rows,dict(feature_sha256=sha(path),fingerprint=str(d['fingerprint']),device='cpu',encoder_executed=False)
