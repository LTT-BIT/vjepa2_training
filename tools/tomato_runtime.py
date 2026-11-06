"""Initialize reproducible tomato runs before any CUDA context or encoder call."""
from __future__ import annotations

import os
import random

import numpy as np

_STATE = None


def initialize_runtime(config):
    global _STATE
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if previous not in (None, ':4096:8', ':16:8'):
        raise RuntimeError('Incompatible CUBLAS_WORKSPACE_CONFIG; set :4096:8 and launch a fresh process')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    initialized = torch.cuda.is_initialized()
    if initialized and (_STATE is None or _STATE['pid'] != os.getpid()):
        raise RuntimeError('CUDA was initialized before tomato deterministic setup; launch a fresh process')
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config.get('device', 'cpu').startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu explicitly')
    _STATE = dict(pid=os.getpid(), seed=seed, device=config.get('device', 'cpu'),
                  cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                  deterministic_algorithms=True, cudnn_benchmark=False, tf32=False,
                  cuda_initialized_before_first_setup=_STATE['cuda_initialized_before_first_setup'] if _STATE else initialized,
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  random_generators=['python', 'numpy', 'torch', 'torch.cuda (manual_seed lazy seeds all devices)'])
    return dict(_STATE)


def hardware_report():
    import torch
    result = dict(_STATE or {})
    result['cuda_available'] = torch.cuda.is_available()
    # Called only AFTER initialization, so these queries cannot preempt deterministic setup.
    if _STATE and _STATE['device'].startswith('cuda'):
        result.update(gpu=torch.cuda.get_device_name(), bf16_supported=torch.cuda.is_bf16_supported())
    return result
