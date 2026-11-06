import os
import subprocess
import sys

from tools import tomato_dataset as ds


def process(code):
    env = dict(os.environ)
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    return subprocess.run([sys.executable, '-c', code], cwd=ds.REPO, env=env, capture_output=True, text=True)


def test_cpu_runtime_sets_environment_before_cuda_and_reproducible_seeds():
    result = process('''
from tools.tomato_runtime import initialize_runtime
import torch, os
assert not torch.cuda.is_initialized()
s = initialize_runtime(dict(device='cpu',seed=17))
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
assert not s['cuda_initialized_before_first_setup']
a=torch.rand(3)
initialize_runtime(dict(device='cpu',seed=17))
assert torch.equal(a,torch.rand(3))
os.environ['CUBLAS_WORKSPACE_CONFIG']='invalid'
try: initialize_runtime(dict(device='cpu',seed=17))
except RuntimeError as e: assert 'fresh process' in str(e)
else: raise AssertionError('invalid cuBLAS accepted')
''')
    assert result.returncode == 0, result.stderr


def test_actual_cuda_setup_precedes_context_and_rejects_late_setup():
    result = process('''
import torch
if not torch.cuda.is_available():
    print('CUDA_UNAVAILABLE'); raise SystemExit(0)
from tools.tomato_runtime import initialize_runtime,hardware_report
initialize_runtime(dict(device='cuda',seed=42))
a=torch.randn(16,16,device='cuda'); b=a@a
initialize_runtime(dict(device='cuda',seed=42))
c=torch.randn(16,16,device='cuda'); d=c@c
assert torch.equal(b,d)
assert hardware_report()['bf16_supported']
print('ACTUAL_CUDA_DETERMINISTIC_OK')
''')
    assert result.returncode == 0, result.stderr
    if 'CUDA_UNAVAILABLE' not in result.stdout:
        assert 'ACTUAL_CUDA_DETERMINISTIC_OK' in result.stdout
    late = process('''
import torch
if not torch.cuda.is_available(): raise SystemExit(0)
torch.zeros(1,device='cuda')
from tools.tomato_runtime import initialize_runtime
try: initialize_runtime(dict(device='cuda',seed=42))
except RuntimeError as e: assert 'fresh process' in str(e)
else: raise AssertionError('late setup accepted')
''')
    assert late.returncode == 0, late.stderr
