"""Read-only loopback video review. Only catalogued video paths are exposed."""
from pathlib import Path
from http.server import ThreadingHTTPServer
import argparse
import mimetypes
from urllib.parse import urlsplit
from . import tomato_dataset as ds
from .serve_tomato_boundary_review import handler, byte_range


def review_handler(root):
    root=Path(root);catalog={r['episode_id']:r for r in ds.read_json(root/'catalog.json')};Base=handler(root)
    class Review(Base):
        def do_GET(self):
            path=urlsplit(self.path).path
            if not path.startswith('/media/'):return super().do_GET()
            try:
                _,_,eid,cam=path.split('/');source=Path(catalog[eid]['videos'][int(cam)]);size=source.stat().st_size
                if cam not in ('0','1','2'):raise ValueError()
                a,b,partial=byte_range(self.headers.get('Range'),size)
            except (KeyError,ValueError,IndexError,FileNotFoundError):self.send_error(404);return
            self.send_response(206 if partial else 200);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(b-a+1))
            if partial:self.send_header('Content-Range',f'bytes {a}-{b}/{size}')
            self.end_headers()
            try:
                with source.open('rb') as f:
                    f.seek(a);remaining=b-a+1
                    while remaining:
                        chunk=f.read(min(65536,remaining))
                        if not chunk:break
                        self.wfile.write(chunk);remaining-=len(chunk)
            except (BrokenPipeError,ConnectionResetError):pass
    return Review


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--port',type=int,default=8770);a=p.parse_args();print(f'Operation review http://127.0.0.1:{a.port}/',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),review_handler(a.directory)).serve_forever()
