"""Stream one selected episode to an atomic, provenance-bound H.264 MP4."""
import json,time,shutil
from pathlib import Path
from .timeline import LABELS,held,MODES
from .catalog import sha,write
COLORS=[(230,135,60),(80,170,70),(50,175,240),(190,100,170)]

def export_video(catalog,result,output,progress=lambda v:None):
    import av,cv2
    import numpy as np
    e=catalog.get(result['episode_id']);mode=result['prediction_mode'];key=result['result_id']
    dest=Path(output)/'ui_exports'/e['episode_id']/mode/key
    if (dest/'render_metadata.json').exists() and (dest/'annotated.mp4').exists():
        receipt=json.loads((dest/'render_metadata.json').read_text())
        if receipt['video_sha256']!=sha(dest/'annotated.mp4'):raise ValueError('Export changed')
        return dict(file=str(dest/'annotated.mp4'),metadata=receipt)
    for rel,h in e['input_sha256'].items():
        if sha(Path(e['dataset_root'])/rel)!=h:raise ValueError('Source changed since training')
    dest.mkdir(parents=True,exist_ok=True);tmp=dest/'annotated.partial.mp4';target=dest/'annotated.mp4'
    inputs=[];output_container=None;start=time.monotonic();count=0
    try:
        inputs=[av.open(str(catalog.media(e['episode_id'],'cam'+str(i)))) for i in range(3)]
        decoders=[c.decode(video=0) for c in inputs]
        output_container=av.open(str(tmp),mode='w');stream=output_container.add_stream('libx264',rate=round(e['fps']))
        stream.width=1152;stream.height=680;stream.pix_fmt='yuv420p';stream.options={'crf':'23','preset':'veryfast'}
        gt=e['annotation']['segments'];heads=MODES[mode];rows=result['rows']
        def text(img,s,x,y,color=(225,230,235),scale=.45):cv2.putText(img,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
        for i in range(e['length']):
            canvas=np.full((680,1152,3),22,dtype=np.uint8)
            for k,decoder in enumerate(decoders):
                frame=next(decoder)
                if frame.pts is None or abs(float(frame.time)-i/e['fps'])>1/e['fps']+.001:raise ValueError('Camera PTS mismatch')
                img=frame.to_ndarray(format='bgr24');h,w=img.shape[:2];ratio=min(384/w,320/h)
                img=cv2.resize(img,(round(w*ratio),round(h*ratio)));x=k*384+(384-img.shape[1])//2;y=(320-img.shape[0])//2
                canvas[y:y+img.shape[0],x:x+img.shape[1]]=img;text(canvas,'cam'+str(k),k*384+10,20)
            row=held(rows,i);truth=next(s['label'] for s in gt if s['start_frame']<=i<s['end_frame'])
            text(canvas,f'{mode} | frame={i} t={i/e["fps"]:.3f}s | GT(now)={truth}',12,345,scale=.6)
            for j,h in enumerate(heads):
                p=row['heads'][h];target_time=p['target_at'];state='NEW' if row['frame_index']==i else 'HELD'
                label=f'{h}'+('(+1.0s)' if h=='future' else '')
                text(canvas,f'{label}: {p["stage"]} {p["confidence"]:.3f} | {state} anchor={p["observed_at"]:.3f} target={target_time:.3f}'+(' TARGET_GT_UNSCORABLE' if not p['target_scorable'] else ''),12,375+j*45)
                text(canvas,' | '.join(f'{k} {v:.3f}' for k,v in p['probabilities'].items()),12,393+j*45)
            tracks=[('GT(now)',[dict(start_s=s['start_frame']/e['fps'],end_s=s['end_frame']/e['fps'],stage=s['label']) for s in gt])]+[(h+('(+1s at original anchor)' if h=='future' else ''),result['tracks'][h]) for h in heads]
            for j,(name,segs) in enumerate(tracks):
                y=500+j*52;text(canvas,name,12,y-7)
                for seg in segs:
                    a=int(seg['start_s']/(e['length']/e['fps'])*1128)+12;b=int(seg['end_s']/(e['length']/e['fps'])*1128)+12
                    cv2.rectangle(canvas,(a,y),(b,y+20),COLORS[LABELS.index(seg['stage'])],-1)
                x=int(i/e['length']*1128)+12;cv2.line(canvas,(x,y-3),(x,y+23),(255,255,255),2)
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas,format='bgr24')):output_container.mux(packet)
            count+=1
            if i%30==0:progress(dict(frame=i,total=e['length']))
        for dec in decoders:
            if next(dec,None) is not None:raise ValueError('Extra source video frames')
        for packet in stream.encode():output_container.mux(packet)
        output_container.close();output_container=None
        with av.open(str(tmp)) as check:
            if sum(1 for _ in check.decode(video=0))!=e['length']:raise ValueError('Incomplete export')
        tmp.replace(target)
        meta=dict(episode_id=e['episode_id'],source_recording_id=e['source_recording_id'],prediction_mode=mode,result_id=key,
            provenance=result['provenance'],source_sha256=e['input_sha256'],frames=count,fps=e['fps'],duration_s=count/e['fps'],
            layout='three letterboxed cameras + GT and selected heads + observation-aligned tracks',elapsed_s=time.monotonic()-start,
            generated_at=time.time(),video_sha256=sha(target))
        (dest/'anchors.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rows));write(dest/'render_metadata.json',meta)
        return dict(file=str(target),metadata=meta)
    except BaseException:
        if output_container:output_container.close()
        tmp.unlink(missing_ok=True);raise
    finally:
        for c in inputs:c.close()
