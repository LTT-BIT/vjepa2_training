"""One real four-view supervised optimizer step, no saved pretrained changes.

Resource feasibility ONLY: not a formal finetuning run or an accuracy result.
"""
import argparse
from functools import partial
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from . import tomato_dataset as ds
from .tomato_phase_model import load_encoder,preprocess
from .tomato_rgb_memory_model import RGBMemoryReader


def run(out,history=1):
    import av,cv2
    if out.exists():raise FileExistsError('Preserve existing probe')
    if history not in (1,8):raise ValueError('Supported resource probes:1 or8 real history windows')
    cfg=ds.read_json('configs/tomato_rgb_memory_train_20261116/protocol.json');m=ds.read_json('runs/tomato/complete100_20261111/dataset_manifest.json');e=next(e for e in m['episodes'] if e['split']=='train');root=Path(cfg['roi_bundle']);mapping=next(r for r in ds.read_json(root/'bundle.json')['mapping'] if (r['source_dataset'],r['source_episode_index'])==(e['dataset_root'],e['episode_index']));ep=mapping['episode_index'];videos=e['videos']+[str(root/f'b_coarse/videos/chunk-000/observation.images.interaction/episode_{ep:06d}.mp4')];clips=[];frame_banks=[];indices=np.array([np.arange(15*j,46+15*j,3) for j in range(history)]);wanted=set(indices.reshape(-1).tolist());last=int(indices[-1,-1])
    for p in videos:
        frames={}
        with av.open(p) as stream:
            for i,frame in enumerate(stream.decode(video=0)):
                if i in wanted:frames[i]=cv2.resize(frame.to_ndarray(format='rgb24'),(256,256),interpolation=cv2.INTER_AREA)
                if i>=last:break
        assert set(frames)==wanted;frame_banks.append(frames)
    clips=[np.stack([frames[int(i)] for i in window]) for window in indices for frames in frame_banks]
    _,q=ds.read_state(e,m['contract']['joint_names'],m['contract']['joint_mode']);z=np.load('runs/tomato/rgb_memory_cache_20261116_r11_v3/'+e['episode_id']+'.npz');stats=np.load('runs/tomato/rgb_memory_cache_20261116_r11_v3/train_state_statistics.npz');target=ds.build_phase_targets(np.array([last]),e['annotation']['segments'],e['length'],e['fps'],1.5)
    from .tomato_runtime import initialize_runtime
    initialize_runtime(dict(device='cuda',seed=42))
    report=dict(scope='One real optimizer step, four views16frames10Hz, batch1; resource feasibility only, NOT a convergence or accuracy result',device=torch.cuda.get_device_name(),raw_inputs_old100_only=True,checkpoint_saved=False,pretrained_files_modified=False,gradient_checkpointing=True,parameter_storage='FP32; autocast BF16; AdamW FP32 state',history_windows=history)
    try:
        encoder,provenance=load_encoder('pretrained/vjepa2-vitl-fpc64-256','cuda');encoder.float().requires_grad_(True).train()
        for layer in encoder.layer:layer.gradient_checkpointing=True;layer._gradient_checkpointing_func=partial(checkpoint,use_reentrant=False)
        net=RGBMemoryReader(stats['mean'],stats['std'],128).cuda().train();params=list(encoder.parameters())+list(net.parameters());opt=torch.optim.AdamW(params,lr=1e-6,foreach=False);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pixels=preprocess(np.stack(clips),'cuda',torch.bfloat16);v=encoder(pixel_values_videos=pixels).last_hidden_state.mean(1).float().reshape(1,history,4,1024);meta=z['metadata'][:history].copy();delta=(np.arange(history)-(history-1))*.5;meta[:,80:96]+=delta[:,None];meta[:,112]=delta;o=net(v,torch.from_numpy(q[indices].reshape(1,history,928)).cuda(),torch.from_numpy(meta[None]).cuda());loss=torch.nn.functional.cross_entropy(o['current_logits'],torch.tensor(target['current_labels'],device='cuda'))+torch.nn.functional.cross_entropy(o['future_logits'],torch.tensor(target['future_labels'],device='cuda'))
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);with_grad=[p for p in encoder.parameters() if p.grad is not None];opt.step();torch.cuda.synchronize();report.update(status='passed',encoder_parameters=sum(p.numel() for p in encoder.parameters()),encoder_gradient_parameters=sum(p.numel() for p in with_grad),head_parameters=sum(p.numel() for p in net.parameters()),loss=float(loss.detach()),optimizer_step_s=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),pretrained=provenance)
    except torch.cuda.OutOfMemoryError as exc:report.update(status='out_of_memory',reason=str(exc),peak_allocated_bytes=torch.cuda.max_memory_allocated());torch.cuda.empty_cache()
    ds.write_json(out,report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--history',type=int,default=1);a=p.parse_args();run(a.out,a.history)
