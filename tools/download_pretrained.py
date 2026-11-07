"""Download public pinned weights in verified HTTP ranges; never execute remote code."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time

import requests
from huggingface_hub import hf_hub_url, get_hf_file_metadata

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent/'pretrained'/'vjepa2-vitl-fpc64-256'
URL = hf_hub_url('facebook/vjepa2-vitl-fpc64-256', 'model.safetensors', revision='b3c1679b7c34d3255ef3547f27c7b226aefab26f')
WEIGHTS_SHA256 = '25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b'
CONFIG_SHA256 = '3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    meta = get_hf_file_metadata(URL, timeout=30)
    if meta.etag != WEIGHTS_SHA256 or meta.commit_hash != 'b3c1679b7c34d3255ef3547f27c7b226aefab26f':
        raise ValueError('Official metadata differs from pinned model identity')
    out.mkdir(parents=True, exist_ok=True)
    config_url = hf_hub_url('facebook/vjepa2-vitl-fpc64-256', 'config.json', revision=meta.commit_hash)
    response = requests.get(config_url, timeout=30)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != CONFIG_SHA256:
        raise ValueError('Official config SHA256 mismatch')
    config_path = out/'config.json'
    if config_path.exists() and config_path.read_bytes() != response.content:
        raise ValueError('Refusing to overwrite a different local config')
    config_path.write_bytes(response.content)
    target = out/'model.safetensors'
    if target.exists():
        with target.open('rb') as stream:
            existing = hashlib.file_digest(stream, 'sha256').hexdigest()
        if existing != WEIGHTS_SHA256:
            raise ValueError('Refusing to overwrite different existing weights')
        (out/'weights_metadata.json').write_text(json.dumps(dict(size=meta.size,sha256=meta.etag,revision=meta.commit_hash),indent=2))
        print('PINNED_PRETRAINED_ALREADY_VERIFIED', flush=True)
        return
    parts = out/'range_parts'
    parts.mkdir(exist_ok=True)
    chunk = 8*1024*1024
    ranges = [(i,min(i+chunk,meta.size)) for i in range(0,meta.size,chunk)]
    def fetch(pair):
        start,end = pair
        path = parts/f'{start:012d}'
        if path.exists() and path.stat().st_size == end-start:
            return path
        for attempt in range(4):
            try:
                # Include range in the query to avoid caches reusing a different range response.
                response = requests.get(URL+f'?download=true&part={start}',headers={'Range':f'bytes={start}-{end-1}'},timeout=(20,120))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end-1}/{meta.size}':
                    raise ValueError('Server did not honor the requested byte range')
                payload = response.content
                if len(payload) != end-start:
                    raise ValueError('Truncated HTTP range')
                tmp=path.with_suffix('.tmp');tmp.write_bytes(payload);tmp.replace(path)
                return path
            except Exception:
                if attempt==3:raise
                time.sleep(min(2**attempt,5))
    started=time.monotonic()
    completed=0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for future in as_completed([pool.submit(fetch,pair) for pair in ranges]):
            path=future.result();completed+=path.stat().st_size
            status=dict(stage='downloading_pretrained',completed_bytes=completed,total_bytes=meta.size,
                        percent=round(completed/meta.size*100,2),elapsed_s=round(time.monotonic()-started,1))
            tmp=out/'status.tmp.json';tmp.write_text(json.dumps(status,indent=2));tmp.replace(out/'status.json')
            print(json.dumps(status),flush=True)
    h=hashlib.sha256()
    tmp=out/'model.safetensors.tmp'
    with tmp.open('wb') as destination:
        for start,end in ranges:
            content=(parts/f'{start:012d}').read_bytes();h.update(content);destination.write(content)
    if h.hexdigest()!=meta.etag:
        raise ValueError('Official weight SHA256 verification failed')
    tmp.replace(target)
    (out/'weights_metadata.json').write_text(json.dumps(dict(size=meta.size,sha256=meta.etag,revision=meta.commit_hash),indent=2))
    for part in parts.iterdir():part.unlink()
    parts.rmdir()
    print('PRETRAINED_DOWNLOAD_COMPLETE_SHA256_VERIFIED',flush=True)


if __name__=='__main__':main()
