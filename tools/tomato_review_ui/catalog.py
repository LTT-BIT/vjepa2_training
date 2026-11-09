import hashlib,json
from pathlib import Path
from .timeline import LABELS,MODES

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def registered(path,root):
    p=Path(path);root=Path(root).absolute()
    if not p.is_absolute():p=root/p
    if not p.is_relative_to(root) or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Media path escapes registered root')
    for q in [p,*p.parents]:
        if q==root.parent:break
        if q.is_symlink():raise ValueError('Symlink media paths are not allowed')
    if not p.is_file():raise FileNotFoundError('Registered media missing')
    return p

class Catalog:
    def __init__(self,root):
        self.root=Path(root).resolve();self.manifests={m:read(self.root/m/'manifest.json') for m in MODES}
        self.hashes={m:sha(self.root/m/'manifest.json') for m in MODES}
        base=self.manifests['joint'];self.episodes={};reference=None
        for mode,m in self.manifests.items():
            if sha(self.root/mode/'split.json')!=m['split_sha256'] or read(self.root/mode/'split.json')!=m['splits']:raise ValueError('Split binding mismatch')
            if m['contract']['labels']!=list(LABELS) or m['prediction']['future_horizon_s']!=1 or m['prediction']['prediction_mode']!=mode:raise ValueError('Task contract mismatch')
            signature=(m['contract'],m['splits'],m['episodes'])
            if reference is None:reference=signature
            elif signature!=reference:raise ValueError('Model input manifests differ')
        for e in base['episodes']:
            if e['episode_id'] in self.episodes:raise ValueError('Duplicate global episode identity')
            s=e['annotation']['segments'];end=0
            for seg in s:
                if seg['start_frame']!=end or seg['end_frame']<=end or seg['label'] not in LABELS or not seg.get('reviewed'):raise ValueError('GT gap/overlap/unreviewed')
                end=seg['end_frame']
            if end!=e['length'] or sha(e['annotation']['path'])!=e['annotation']['sha256'] or read(e['annotation']['path'])['segments']!=s:raise ValueError('Locked GT changed')
            if len(e['videos'])!=3:raise ValueError('Missing camera')
            if any(v.get('max_pts_error_s',0)>1/e['fps'] for v in e.get('media_validation',{}).get('media',[])):raise ValueError('Camera PTS mismatch; original time contract unsupported')
            for p in e['videos']:registered(p,e['dataset_root'])
            self.episodes[e['episode_id']]=e
    def get(self,eid):
        if eid not in self.episodes:raise KeyError('Unknown registered episode')
        return self.episodes[eid]
    def listing(self,split='',q='',package=''):
        result=[]
        for e in self.episodes.values():
            pkg=str(Path(e['dataset_root']).parent.name)+'/'+Path(e['dataset_root']).name
            if split and e['split']!=split:continue
            if package and package!=pkg:continue
            if q.lower() not in f"{pkg} {e['episode_index']} {e['source_recording_id']} {e['episode_id']}".lower():continue
            result.append(dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],episode_index=e['episode_index'],split=e['split'],package=pkg,
                length=e['length'],fps=e['fps'],duration_s=e['length']/e['fps'],segments=len(e['annotation']['segments']),
                modes={m:(self.root/m/'best_head.pt').is_file() for m in MODES}))
        return result
    def detail(self,eid):
        e=self.get(eid)
        return dict(**next(v for v in self.listing() if v['episode_id']==eid),gt=e['annotation']['segments'],
            annotation_sha256=e['annotation']['sha256'],revision=e['annotation']['revision'],
            cameras={cam:f'/api/media/{eid}/{cam}' for cam in ['cam0','cam1','cam2']},
            media_validation=e['media_validation'])
    def media(self,eid,cam):
        if cam not in ['cam0','cam1','cam2']:raise KeyError('Invalid camera')
        e=self.get(eid);return registered(e['videos'][int(cam[-1])],e['dataset_root'])
