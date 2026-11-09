"""Loopback-only review server with byte ranges required for frame-accurate seeking."""
import argparse
import mimetypes
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit


def byte_range(header,size):
    if header is None:return 0,size-1,False
    if not header.startswith('bytes=') or ',' in header:raise ValueError('Only one byte range supported')
    first,last=header[6:].split('-',1)
    if not first:
        count=int(last)
        if count<=0:raise ValueError('Invalid suffix')
        return max(0,size-count),size-1,True
    start=int(first);end=min(size-1,int(last)) if last else size-1
    if not 0<=start<=end<size:raise ValueError('Unsatisfiable range')
    return start,end,True


def handler(root):
    root=Path(root).resolve()
    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=(root/unquote(urlsplit(self.path).path).lstrip('/')).resolve()
            if not path.is_relative_to(root):self.send_error(403);return
            if path.is_dir():path=path/'index.html'
            if not path.is_file():self.send_error(404);return
            size=path.stat().st_size
            try:start,end,partial=byte_range(self.headers.get('Range'),size)
            except (ValueError,TypeError):
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
            self.send_response(206 if partial else 200);self.send_header('Content-Type',mimetypes.guess_type(path)[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1));self.send_header('Cache-Control','no-cache')
            if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with path.open('rb') as f:
                    f.seek(start);remaining=end-start+1
                    while remaining:
                        block=f.read(min(remaining,65536))
                        if not block:break
                        self.wfile.write(block);remaining-=len(block)
            except (BrokenPipeError,ConnectionResetError):pass
    return ReviewHandler


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--directory',default='runs/tomato/boundary_review_20261109/review_package');p.add_argument('--port',type=int,default=8768);a=p.parse_args()
    print(f'Review package http://127.0.0.1:{a.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),handler(a.directory)).serve_forever()
