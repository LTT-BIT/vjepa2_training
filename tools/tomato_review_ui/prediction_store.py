import json,threading
from pathlib import Path
from .catalog import read,sha,write
from .timeline import validate_rows,tracks

class PredictionRepository:
    def __init__(self,catalog,output):
        self.catalog=catalog;self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        for e in catalog.episodes.values():
            if self.output.is_relative_to(Path(e['dataset_root']).resolve()):raise ValueError('UI output must be outside sources')
        if self.output==catalog.root or any(self.output.is_relative_to(catalog.root/m) for m in ['current_only','future_only','joint']):raise ValueError('Output must be a dedicated UI directory')
        self.lock=threading.Lock();self.results={}
        receipt=catalog.root/'analysis/original_result_sha256.json'
        self.receipt=read(receipt) if receipt.exists() else {}
    def recognize(self,eid,mode,regenerate=False):
        from .timeline import MODES
        if mode not in MODES:raise ValueError('Unknown mode')
        with self.lock:
            e=self.catalog.get(eid);run=self.catalog.root/mode;m=self.catalog.manifests[mode]
            if sha(run/'manifest.json')!=self.catalog.hashes[mode]:raise ValueError('Manifest changed after catalog opened')
            checkpoint=sha(run/'best_head.pt');manifest=sha(run/'manifest.json');split=sha(run/'split.json')
            expected=next((v for k,v in self.receipt.items() if str(run/'best_head.pt').endswith(k)),None)
            if expected is not None and checkpoint!=expected:raise ValueError('Locked checkpoint changed')
            for rel,expected in e['input_sha256'].items():
                if sha(Path(e['dataset_root'])/rel)!=expected:raise ValueError('Registered source changed')
            if sha(e['annotation']['path'])!=e['annotation']['sha256']:raise ValueError('Locked annotation changed')
            feature_sha=None
            index_path=run/'feature_index.json'
            if index_path.exists():
                index=read(index_path)
                fp=next((Path(v['path']) for v in index['episodes'] if v['episode_id']==eid),None)
                if fp is not None and fp.exists():feature_sha=sha(fp)
            if split!=m['split_sha256']:raise ValueError('Split changed')
            path=run/'predictions.jsonl';prediction_sha=sha(path) if path.exists() else None
            key=sha_bytes(dict(eid=eid,mode=mode,checkpoint=checkpoint,manifest=manifest,split=split,predictions=prediction_sha,feature_sha=feature_sha,regenerate=regenerate))
            if key in self.results:return self.results[key]
            target=self.output/'predictions'/key/'result.json'
            if target.exists():
                value=read(target)
                if value['provenance']['key']!=key or sha(target)!=read(target.with_suffix('.sha.json'))['sha256']:raise ValueError('Derived result receipt mismatch')
                validate_rows(value['raw_rows'],e,mode);self.results[key]=value;return value
            source='single_episode_head';raw=[];provenance={}
            # Existing audit receipts explicitly bind frozen predictions and checkpoints.
            def trusted(p):
                return any(str(p).endswith(k) and sha(p)==v for k,v in self.receipt.items())
            if path.exists() and not regenerate and trusted(path) and trusted(run/'best_head.pt') and trusted(run/'manifest.json'):
                raw=[json.loads(line) for line in path.open() if json.loads(line)['episode_id']==eid]
                try:validate_rows(raw,e,mode);source='frozen_predictions'
                except (ValueError,KeyError):raw=[]
            if not raw:
                from .recognize import infer
                raw,provenance=infer(self.catalog,eid,mode)
            rows=validate_rows(raw,e,mode)
            value=dict(result_id=key,episode_id=eid,prediction_mode=mode,horizon_s=1.,source=source,rows=rows,
                tracks=tracks(rows,e,mode),raw_rows=raw,provenance=dict(**provenance,key=key,checkpoint_sha256=checkpoint,
                    manifest_sha256=manifest,split_sha256=split,predictions_sha256=prediction_sha,annotation_sha256=e['annotation']['sha256']))
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.tmp');write(temp,value);temp.replace(target);write(target.with_suffix('.sha.json'),dict(sha256=sha(target)))
            self.results[key]=value;return value

def sha_bytes(v):
    import hashlib
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
