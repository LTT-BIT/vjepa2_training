from datetime import datetime,timezone
from .handshake import replay,validate
from .prediction_store import sha_bytes
from .catalog import write,read,sha
from pathlib import Path

def simulate(app,rid,config):
    r=app.store.results[rid]
    if r['prediction_mode']!='joint' or r.get('prediction_hz',2)!=2:raise ValueError('本轮握手仅支持已识别的 Joint 2Hz；不伪造 Current ACK 或 5Hz')
    c=validate(config);engine_sha=sha(Path(__file__).with_name('handshake.py'));key=sha_bytes(dict(policy='handshake-v2',engine_sha=engine_sha,result_id=rid,rows=r['rows'],config=c))
    directory=app.store.output/'handshake'/key;path=directory/'simulation.json'
    if path.exists():return read(path)
    e=app.catalog.get(r['episode_id']);value=replay(r['rows'],c,e['length']/e['fps'])
    value.update(simulation_id=key,result_id=rid,episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],
        provenance={**r['provenance'],'policy_source_sha256':engine_sha},config_sha256=sha_bytes(c),generated_at=datetime.now(timezone.utc).isoformat())
    directory.mkdir(parents=True,exist_ok=True);write(path,value);write(directory/'config.lock.json',dict(config=c,config_sha256=value['config_sha256'],provenance=r['provenance'],result_id=rid))
    (directory/'events.jsonl').write_text(''.join(__import__('json').dumps(v)+'\n' for v in value['events']))
    return value
