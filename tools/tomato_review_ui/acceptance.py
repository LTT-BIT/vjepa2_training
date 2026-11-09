"""Explicit real-data UI acceptance, including TWO selected-episode exports.

Not invoked by the server or page startup. Requires optional playwright+Chrome.
"""
import argparse,json,time,urllib.request,urllib.error,http.cookiejar
from pathlib import Path
from .catalog import read,sha,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:8770');a=p.parse_args()
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    original={str(p):sha(p) for m in ['current_only','future_only','joint'] for p in (a.run_root/m).rglob('*') if p.is_file()}
    jar=http.cookiejar.CookieJar();opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    def api(path,body=None,headers=None):
        req=urllib.request.Request(a.url+path,json.dumps(body).encode() if body is not None else None,headers or {})
        with opener.open(req,timeout=120) as response:return json.load(response)
    token=api('/api/session')['token'];headers={'Content-Type':'application/json','X-Review-Token':token,'Origin':a.url}
    eps=api('/api/episodes');seven=min((e for e in eps if e['segments']==7),key=lambda e:e['length']);three=next(e for e in eps if e['segments']==3)
    evidence=dict(episodes=len(eps),seven=seven,three=three,predictions={},exports={},http={},browser={})
    def job(path,body):
        jid=api(path,body,headers)['job_id'];start=time.monotonic()
        while True:
            j=api('/api/jobs/'+jid)
            if j['status']=='failed':raise RuntimeError(j['error'])
            if j['status']=='complete':return j,time.monotonic()-start
            if time.monotonic()-start>600:raise TimeoutError(jid)
            time.sleep(.2)
    result_ids={}
    for mode in ['future_only','joint','current_only']:
        j,t=job('/api/predictions/recognize',dict(episode_id=seven['episode_id'],prediction_mode=mode));r=api('/api/predictions/'+j['result_id']);result_ids[mode]=r['result_id']
        raw=[json.loads(l) for l in (a.run_root/mode/'predictions.jsonl').read_text().splitlines() if json.loads(l)['episode_id']==seven['episode_id']]
        for k in [0,len(raw)//2,len(raw)-1]:
            for head,headrow in r['rows'][k]['heads'].items():assert headrow['probabilities']==raw[k][head+'_probabilities']
        if mode!='current_only':assert not r['rows'][-1]['heads']['future']['target_scorable']
        evidence['predictions'][mode]=dict(source=j['source'],seconds=t,anchors=len(r['rows']),result_id=r['result_id'])
    # Explicit acceptance-only regeneration exercises the CPU fallback once, no Encoder.
    j,t=job('/api/predictions/recognize',dict(episode_id=three['episode_id'],prediction_mode='future_only',regenerate=True))
    r=api('/api/predictions/'+j['result_id']);assert j['source']=='single_episode_head'
    evidence['fallback']=dict(seconds=t,provenance=r['provenance'],anchors=len(r['rows']))
    for range_header in [None,'bytes=0-31','bytes=-16','bytes=99999999999-','bytes=0-1,4-5']:
        req=urllib.request.Request(a.url+f'/api/media/{seven["episode_id"]}/cam0',headers={'Range':range_header} if range_header else {})
        try:
            with opener.open(req) as response:evidence['http'][str(range_header)]=dict(status=response.status,content_range=response.headers.get('Content-Range'),length=response.headers.get('Content-Length'));response.read(32)
        except urllib.error.HTTPError as e:evidence['http'][str(range_header)]=dict(status=e.code)
    assert [v['status'] for v in evidence['http'].values()]==[200,206,206,416,416]
    for path in [f'/api/media/{seven["episode_id"]}/../etc/passwd',f'/api/media/{seven["episode_id"]}/cam9']:
        try:api(path);raise AssertionError('Unsafe path accepted')
        except urllib.error.HTTPError as e:assert e.code==404
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=browser.new_page(viewport={'width':1560,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(a.url)
        for sample in [seven,three]:
            page.locator('#search').fill(sample['source_recording_id']);page.wait_for_timeout(350);page.locator('.episode').click()
            page.wait_for_function('episode !== null && videos.every(v=>v.readyState>=2)')
            assert page.locator('.track').count()==1
            for mode in ['future_only','joint']:
                page.locator('#mode').select_option(mode);assert page.locator('#export').is_disabled();page.locator('#recognize').click()
                page.wait_for_function('result!==null',timeout=120000)
                assert page.locator('.track').count()==(2 if mode=='future_only' else 3)
                page.evaluate('seek(37)');page.wait_for_timeout(300)
                times=page.evaluate('videos.map(v=>v.currentTime)');assert max(times)-min(times)<1/sample['fps']
                assert page.evaluate('held().frame_index')==30
                assert page.evaluate("held().heads.future.target_at")==2.
                page.locator('#next').click();assert page.evaluate('frame')==38
                page.locator('#play').click();page.wait_for_timeout(1000);page.locator('#play').click()
                evidence['browser'][sample['episode_id']+'_'+mode]=dict(times=times,frame_after_play=page.evaluate('frame'))
                page.screenshot(path=str(out/(sample['episode_id']+'_'+mode+'.png')),full_page=True)
        evidence['browser']['errors']=errors;assert not errors;browser.close()
    import av
    for mode in ['future_only','joint']:
        j,t=job('/api/exports/video',dict(result_id=result_ids[mode]));meta=j['metadata']
        with opener.open(a.url+j['download_url']) as response:
            dest=out/f'{mode}.mp4'
            with dest.open('wb') as f:
                while True:
                    b=response.read(1024*1024)
                    if not b:break
                    f.write(b)
        with av.open(str(dest)) as container:
            decoded=0
            for frame in container.decode(video=0):
                if decoded in [0,37,seven['length']-1]:frame.to_image().save(out/f'{mode}_frame{decoded}.png')
                decoded+=1
        assert decoded==seven['length'] and sha(dest)==meta['video_sha256']
        evidence['exports'][mode]=dict(seconds=t,frames=decoded,metadata=meta,download=str(dest))
    evidence['protected_files']=len(original);evidence['originals_unchanged']=all(sha(p)==h for p,h in original.items());assert evidence['originals_unchanged']
    write(out/'acceptance.json',evidence);print(out/'acceptance.json')
if __name__=='__main__':main()
