import copy,json
from pathlib import Path
import pytest
from tools.tomato_review_ui.timeline import validate_rows,held,tracks
from tools.tomato_review_ui.server import byte_range
from tools.tomato_review_ui.catalog import registered


def fixture():
    e=dict(episode_id='e',source_recording_id='s',split='val',fps=30,length=30,anchors=[0,15])
    rows=[dict(episode_id='e',source_recording_id='s',split='val',prediction_mode='future_only',frame_index=i,timestamp_s=i/30,
        last_input_frame=i,input_indices=[i]*16,future_probabilities=dict(pickup=.1,harvest=.7,**{'return':.1},adjust=.1),future_valid=False) for i in e['anchors']]
    return e,rows


def test_held_keeps_original_target_and_unscorable_tail_visible():
    e,raw=fixture();rows=validate_rows(raw,e,'future_only')
    assert held(rows,14)['heads']['future']['target_at']==1.
    assert not held(rows,29)['heads']['future']['target_scorable']
    assert list(tracks(rows,e,'future_only'))==['future']
    assert tracks(rows,e,'future_only')['future'][-1]['end_s']==1.


@pytest.mark.parametrize('mutation',['missing','duplicate','source','mode','nan','future_input','sum'])
def test_bad_predictions_rejected(mutation):
    e,r=fixture()
    if mutation=='missing':r.pop()
    if mutation=='duplicate':r[1]=r[0]
    if mutation=='source':r[0]['source_recording_id']='other'
    if mutation=='mode':r[0]['prediction_mode']='joint'
    if mutation=='nan':r[0]['future_probabilities']['pickup']=float('nan')
    if mutation=='future_input':r[0]['last_input_frame']=1
    if mutation=='sum':r[0]['future_probabilities']['pickup']=.9
    with pytest.raises((ValueError,KeyError)):validate_rows(r,e,'future_only')


@pytest.mark.parametrize('header,result',[(None,(0,99,200)),('bytes=0-9',(0,9,206)),('bytes=90-',(90,99,206)),('bytes=-10',(90,99,206)),('bytes=0-999',(0,99,206))])
def test_byte_ranges(header,result):assert byte_range(header,100)==result


@pytest.mark.parametrize('header',['bytes=100-','bytes=5-1','bytes=-0','bytes=a-b','bytes=0-1,3-4','items=0-1'])
def test_invalid_ranges(header):
    with pytest.raises(ValueError):byte_range(header,100)


def test_media_paths_reject_escape_and_symlink(tmp_path):
    root=tmp_path/'root';root.mkdir();video=root/'v.mp4';video.write_bytes(b'fake')
    assert registered(video,root)==video
    outside=tmp_path/'outside.mp4';outside.write_bytes(b'other');(root/'link.mp4').symlink_to(outside)
    with pytest.raises(ValueError):registered(root/'link.mp4',root)
    with pytest.raises(ValueError):registered(outside,root)


def test_old_alias_never_substitutes_future_head():
    e,r=fixture()
    for row in r:row.pop('future_probabilities');row['stage']='harvest'
    with pytest.raises(KeyError):validate_rows(r,e,'future_only')


def make_catalog(tmp_path,mutation=None):
    from tools.tomato_review_ui.catalog import write,sha,Catalog
    es=[]
    for i in range(2):
        root=tmp_path/f'data{i}';root.mkdir();vs=[]
        for k in range(3):
            p=root/f'cam{k}.mp4';p.write_bytes(b'synthetic');vs.append(str(p))
        segments=[dict(start_frame=0,end_frame=30,label='pickup',reviewed=True)]
        ann=root/'ann.json';write(ann,dict(segments=segments))
        es.append(dict(episode_id=f'e{i}',source_recording_id=f's{i}',episode_index=0,split='val',dataset_root=str(root),fps=30,length=30,anchors=[0,15],videos=vs,
            annotation=dict(path=str(ann),sha256=sha(ann),segments=segments,revision=1),media_validation={}))
    if mutation=='gap':es[0]['annotation']['segments'][0]['start_frame']=1
    if mutation=='overlap':es[0]['annotation']['segments'].append(dict(start_frame=10,end_frame=30,label='return',reviewed=True))
    if mutation=='duplicate':es[1]['episode_id']='e0'
    if mutation=='camera':es[0]['videos'].pop()
    for mode in ['current_only','future_only','joint']:
        d=tmp_path/mode;d.mkdir();write(d/'split.json',dict(val=['s0','s1']))
        write(d/'manifest.json',dict(contract=dict(labels=['pickup','harvest','return','adjust']),prediction=dict(prediction_mode=mode,future_horizon_s=1),splits=dict(val=['s0','s1']),split_sha256=sha(d/'split.json'),episodes=es))
    return Catalog(tmp_path)


def test_global_ids_allow_same_episode_index(tmp_path):
    c=make_catalog(tmp_path);assert len(c.listing())==2 and len(c.listing(q='s0'))==1
    assert c.get('e0')['episode_index']==c.get('e1')['episode_index']


@pytest.mark.parametrize('mutation',['gap','overlap','duplicate','camera'])
def test_invalid_catalog_fail_closed(tmp_path,mutation):
    with pytest.raises(ValueError):make_catalog(tmp_path,mutation)


def test_missing_checkpoint_reports_failure_without_source_write(tmp_path):
    from tools.tomato_review_ui.prediction_store import PredictionRepository
    c=make_catalog(tmp_path);s=PredictionRepository(c,tmp_path/'ui')
    with pytest.raises(FileNotFoundError):s.recognize('e0','future_only')
    assert not list((tmp_path/'ui').rglob('result.json'))


def test_export_interruption_never_marks_complete(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from tools.tomato_review_ui.video_export import export_video
    def fail(*a,**k):raise RuntimeError('injected decoder interruption')
    monkeypatch.setitem(sys.modules,'av',SimpleNamespace(open=fail))
    e=dict(episode_id='e',input_sha256={})
    catalog=SimpleNamespace(get=lambda eid:e,media=lambda eid,cam:tmp_path/'x')
    with pytest.raises(RuntimeError):export_video(catalog,dict(episode_id='e',prediction_mode='future_only',result_id='id'),tmp_path/'output')
    assert not list((tmp_path/'output').rglob('render_metadata.json'))
    assert not list((tmp_path/'output').rglob('*.mp4'))


def test_http_auth_range_and_job_ownership(tmp_path):
    import threading,urllib.request,urllib.error,http.cookiejar,time
    from http.server import ThreadingHTTPServer
    from tools.tomato_review_ui.server import App,handler
    make_catalog(tmp_path);app=App(tmp_path,tmp_path/'ui')
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(app));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    jar=http.cookiejar.CookieJar();client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    try:
        with client.open(base+'/api/session') as r:token=json.load(r)['token']
        def post(headers):return urllib.request.Request(base+'/api/predictions/recognize',json.dumps(dict(episode_id='e0',prediction_mode='future_only')).encode(),headers)
        with pytest.raises(urllib.error.HTTPError) as exc:client.open(post({}))
        assert exc.value.code==403
        headers={'Content-Type':'application/json','X-Review-Token':token,'Origin':base}
        with client.open(post(headers)) as response:jid=json.load(response)['job_id']
        with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(base+'/api/jobs/'+jid)
        assert exc.value.code==403
        with client.open(urllib.request.Request(base+'/api/media/e0/cam0',headers={'Range':'bytes=0-1'})) as response:assert response.status==206 and response.read()==b'sy'
        with pytest.raises(urllib.error.HTTPError) as exc:client.open(urllib.request.Request(base+'/api/media/e0/cam0',headers={'Range':'bytes=999-'}))
        assert exc.value.code==416
        with pytest.raises(urllib.error.HTTPError) as exc:client.open(base+'/api/media/e0/cam9')
        assert exc.value.code==404
    finally:server.shutdown();server.server_close();app.pool.shutdown();thread.join()


def test_job_dedup_invalidates_when_checkpoint_changes(tmp_path):
    import time
    from tools.tomato_review_ui.server import App
    make_catalog(tmp_path);app=App(tmp_path,tmp_path/'ui')
    app.store.recognize=lambda *a,**kw:dict(result_id='fake',source='fixture')
    try:
        data=dict(episode_id='e0',prediction_mode='future_only')
        first=app.submit('owner','recognize',data)
        deadline=time.monotonic()+3
        while app.jobs[first]['status']!='complete' and time.monotonic()<deadline:time.sleep(.01)
        assert app.submit('owner','recognize',data)==first
        (tmp_path/'future_only/best_head.pt').write_bytes(b'changed-model')
        assert app.submit('owner','recognize',data)!=first
    finally:app.pool.shutdown()
