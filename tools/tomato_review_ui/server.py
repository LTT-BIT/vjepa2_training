"""Loopback-only HTTP server, isolated background jobs and registered media ranges."""
import argparse,json,mimetypes,re,secrets,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse,parse_qs,unquote
from http.cookies import SimpleCookie
from .catalog import Catalog,sha,read
from .prediction_store import PredictionRepository,sha_bytes
from .video_export import export_video


def byte_range(value,size):
    if not value:return 0,size-1,200
    m=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
    if not m or not any(m.groups()) or size<=0:raise ValueError('Invalid Range')
    a,b=m.groups()
    if not a:
        n=int(b)
        if not n:raise ValueError('Empty suffix')
        start=max(0,size-n);end=size-1
    else:start=int(a);end=min(int(b),size-1) if b else size-1
    if start>=size or start>end:raise ValueError('Unsatisfiable Range')
    return start,end,206

class App:
    def __init__(self,root,output):
        self.catalog=Catalog(root);self.store=PredictionRepository(self.catalog,output)
        self.pool=ThreadPoolExecutor(max_workers=1);self.jobs={};self.keys={};self.permissions={};self.sessions=set();self.lock=threading.Lock()
    def submit(self,owner,kind,data):
        eid=data.get('episode_id');mode=data.get('prediction_mode');rid=data.get('result_id')
        revision=None
        if kind=='recognize':
            from .timeline import MODES
            if mode not in MODES:raise ValueError('Unknown mode')
            e=self.catalog.get(eid);run=self.catalog.root/mode
            paths=[run/n for n in ['best_head.pt','manifest.json','split.json','predictions.jsonl','feature_index.json']]
            paths += [Path(e['annotation']['path'])]+[Path(e['dataset_root'])/n for n in e.get('input_sha256',{})]
            if (run/'feature_index.json').exists():
                item=next((v for v in read(run/'feature_index.json')['episodes'] if v['episode_id']==eid),None)
                if item:paths.append(Path(item['path']))
            revision=sha_bytes({str(p):sha(p) if p.is_file() else None for p in paths})
        elif rid not in self.permissions.get(owner,set()):raise PermissionError('Result not owned by session')
        key=(owner,kind,eid,mode,rid,bool(data.get('regenerate',False)),revision)
        with self.lock:
            if key in self.keys and self.jobs[self.keys[key]]['status']!='failed':return self.keys[key]
            jid=secrets.token_hex(16);self.keys[key]=jid;self.jobs[jid]=dict(job_id=jid,owner=owner,status='queued',kind=kind)
        def work():
            job=self.jobs[jid];job['status']='running'
            try:
                if kind=='recognize':
                    value=self.store.recognize(eid,mode,data.get('regenerate',False));rid2=value['result_id']
                    self.permissions.setdefault(owner,set()).add(rid2);job['result_id']=rid2;job['source']=value['source']
                else:
                    result=self.store.results[rid]
                    job['export']=export_video(self.catalog,result,self.store.output,lambda v:job.update(progress=v))
                job['status']='complete'
            except Exception as exc:job.update(status='failed',error=str(exc))
        self.pool.submit(work);return jid


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def session(self):
            cookie=SimpleCookie(self.headers.get('Cookie',''));value=cookie.get('review_session')
            return value.value if value and value.value in app.sessions else None
        def reply(self,v,code=200,headers=None):
            body=json.dumps(v,ensure_ascii=False,allow_nan=False).encode();self.send_response(code)
            for k,val in (headers or {}).items():self.send_header(k,val)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def file(self,path):
            size=path.stat().st_size
            try:a,b,status=byte_range(self.headers.get('Range'),size)
            except ValueError:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
            self.send_response(status);self.send_header('Content-Type',mimetypes.guess_type(path)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(b-a+1));self.send_header('X-Content-Type-Options','nosniff')
            if status==206:self.send_header('Content-Range',f'bytes {a}-{b}/{size}')
            self.end_headers()
            with path.open('rb') as f:
                f.seek(a);remaining=b-a+1
                while remaining:
                    data=f.read(min(262144,remaining))
                    if not data:break
                    self.wfile.write(data);remaining-=len(data)
        def do_GET(self):
            try:
                host=self.headers.get('Host','')
                if host not in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']:raise PermissionError('Loopback Host required')
                parsed=urlparse(self.path);path=unquote(parsed.path);parts=path.strip('/').split('/');owner=self.session()
                if path=='/api/session':
                    token=owner or secrets.token_hex(24);app.sessions.add(token)
                    return self.reply(dict(token=token),headers={'Set-Cookie':f'review_session={token}; HttpOnly; SameSite=Strict; Path=/'})
                if path=='/api/episodes':
                    q=parse_qs(parsed.query);return self.reply(app.catalog.listing(q.get('split',[''])[0],q.get('q',[''])[0],q.get('package',[''])[0]))
                if len(parts)==3 and parts[:2]==['api','episodes']:return self.reply(app.catalog.detail(parts[2]))
                if len(parts)==4 and parts[:2]==['api','media']:return self.file(app.catalog.media(parts[2],parts[3]))
                if len(parts)==3 and parts[:2]==['api','predictions']:
                    if parts[2] not in app.permissions.get(owner,set()):raise PermissionError('Unknown result for session')
                    v=app.store.results[parts[2]];return self.reply({k:x for k,x in v.items() if k!='raw_rows'})
                if len(parts)>=3 and parts[0]=='api' and parts[1] in ['jobs','exports']:
                    job=app.jobs.get(parts[2])
                    if not job or job['owner']!=owner:raise PermissionError('Job belongs to another session')
                    if len(parts)==4 and parts[3]=='download' and job['status']=='complete' and 'export' in job:return self.file(Path(job['export']['file']))
                    return self.reply({k:v for k,v in job.items() if k not in ['owner','export']}|({'metadata':job['export']['metadata'],'download_url':f'/api/exports/{parts[2]}/download'} if 'export' in job else {}))
                names={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
                if path in names:return self.file(Path(__file__).parent/'static'/names[path])
                self.reply(dict(error='Not found'),404)
            except (KeyError,FileNotFoundError) as exc:self.reply(dict(error=str(exc)),404)
            except PermissionError as exc:self.reply(dict(error=str(exc)),403)
            except (ValueError,TypeError) as exc:self.reply(dict(error=str(exc)),400)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            try:
                owner=self.session();origin=self.headers.get('Origin')
                if self.headers.get('Host') not in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']:raise PermissionError('Loopback Host required')
                if not owner or self.headers.get('X-Review-Token')!=owner or origin!=f'http://{self.headers.get("Host")}':raise PermissionError('Same-origin session token required')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=4096:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(n))
                if not isinstance(data,dict):raise ValueError('JSON object required')
                if self.path=='/api/predictions/recognize':
                    if set(data)-{'episode_id','prediction_mode','regenerate'}:raise ValueError('Unexpected input fields')
                    jid=app.submit(owner,'recognize',data)
                elif self.path=='/api/exports/video':
                    if set(data)!={'result_id'}:raise ValueError('Only result_id is accepted')
                    jid=app.submit(owner,'export',data)
                else:return self.reply(dict(error='Not found'),404)
                self.reply(dict(job_id=jid),202)
            except PermissionError as exc:self.reply(dict(error=str(exc)),403)
            except (ValueError,TypeError,KeyError) as exc:self.reply(dict(error=str(exc)),400)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    p.add_argument('--host',choices=['127.0.0.1','localhost'],default='127.0.0.1');p.add_argument('--port',type=int,default=8770)
    a=p.parse_args();app=App(a.run_root,a.output_dir or a.run_root/'review_ui');server=ThreadingHTTPServer((a.host,a.port),handler(app))
    print(f'Read-only review: http://{a.host}:{a.port}',flush=True)
    try:server.serve_forever()
    finally:app.pool.shutdown(wait=True);server.server_close()

if __name__=='__main__':main()
