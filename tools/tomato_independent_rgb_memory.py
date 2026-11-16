"""Independent trainable readouts sharing only frozen input features/statistics."""
from torch import nn
from .tomato_rgb_memory_model import RGBMemoryReader

ARCHITECTURE='old100_independent_rgb_memory_v1'


class SingleRGBReader(RGBMemoryReader):
    def __init__(self, mean, std, hidden=128, memory_scope='all', target='current'):
        if target not in ('current','future'):
            raise ValueError('Single supervised target required')
        super().__init__(mean,std,hidden,memory_scope)
        self.target=target
        # Remove the unused output's parameters entirely, preserving matched
        # trunk initialization with the existing joint reader.
        if target=='current':self.future=nn.Identity()
        else:self.current=nn.Identity()

    def forward(self, **inputs):
        return super().forward(**inputs)[self.target+'_logits']


class IndependentRGBReader(nn.Module):
    def __init__(self,current_branch,future_branch):
        super().__init__()
        if current_branch.target!='current' or future_branch.target!='future':
            raise ValueError('Current and future readers required')
        if {id(p) for p in current_branch.parameters()} & {id(p) for p in future_branch.parameters()}:
            raise ValueError('Independent readers cannot share trainable parameters')
        self.current_branch=current_branch;self.future_branch=future_branch

    def forward(self, **inputs):
        return dict(current_logits=self.current_branch(**inputs),
                    future_logits=self.future_branch(**inputs))


def restore_reader(saved):
    from .tomato_rgb_memory_training import memory_scope
    args=(saved['state_mean'],saved['state_std'],saved['protocol']['hidden'],memory_scope(saved['method']))
    net=IndependentRGBReader(SingleRGBReader(*args,target='current'),SingleRGBReader(*args,target='future'))
    net.load_state_dict(saved['head'])
    return net
