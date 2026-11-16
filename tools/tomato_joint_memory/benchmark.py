"""Cached readout latency is distinct from unmeasured online Encoder latency."""
import time
import numpy as np
import torch
from .cache_reader import CausalBuffer
from .train import TensorData

@torch.inference_mode()
def benchmark(model,saved,entry,c,group,beliefs=None):
    data=TensorData([entry],c,group,saved['stats'],beliefs);ii=slice(min(10,len(data.x)-1),min(10,len(data.x)-1)+1)
    def sync():
        if c['device'].startswith('cuda'):torch.cuda.synchronize()
    for _ in range(20):data.forward(model,ii)
    sync();ms=[]
    for _ in range(150):
        sync();start=time.perf_counter();data.forward(model,ii);sync();ms.append((time.perf_counter()-start)*1000)
    n=c['memory_lengths'].get(group,1);buffer=CausalBuffer(n,c['gap_reset_s']);buffer.reset('benchmark','benchmark','locked');bm=[]
    for i in range(100):
        start=time.perf_counter();buffer.push(entry['features'][i%len(entry['features'])],i*.5,'benchmark','benchmark','locked');bm.append((time.perf_counter()-start)*1000)
    return dict(cached_readout_ms={f'p{p}':float(np.percentile(ms,p)) for p in [50,95,99]},cpu_buffer_ms={f'p{p}':float(np.percentile(bm,p)) for p in [50,95,99]},raw_history_capacity_bytes=n*4000*4,aux_history_capacity_bytes=n*(58 if group=='K1' else 9 if group=='P1' else 0)*4,memory_length=n,readout_benchmark='batch=1 GPU FP32 resident frozen feature; normalization/transfer/Encoder/decode not included',encoder_new_calls=0,online_encoder_plus_memory='NOT_MEASURED: cached campaign deliberately performs no new Encoder extraction',pi05_concurrency='NOT_MEASURED',belief_teacher_readout='add fixed B0 cached_readout to P1; feature Encoder shared, probabilities must be produced online, not free')
