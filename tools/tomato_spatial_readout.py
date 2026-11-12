"""Matched attention readout: pooled vs coarse spatial tokens, no task memory."""
import torch
from torch import nn


def spatial_pool(tokens,frames=16,image=256,patch=16,tubelet=2,grid=4):
    b,n,d=tokens.shape;t=frames//tubelet;s=image//patch
    if n!=t*s*s or s%grid:raise ValueError('Verified Conv3D flattened temporal/spatial token layout required')
    return tokens.float().reshape(b,t,grid,s//grid,grid,s//grid,d).mean((1,3,5)).reshape(b,grid*grid,d)


class SpatialReadout(nn.Module):
    def __init__(self):
        super().__init__();self.visual=nn.Linear(1024,64);self.state=nn.Linear(928,64);self.position=nn.Parameter(torch.zeros(1,48,64));self.attention=nn.MultiheadAttention(64,4,batch_first=True);self.norm=nn.LayerNorm(64);self.classifier=nn.Linear(64,4)

    def forward(self,tokens,positions,method):
        if tokens.ndim!=4 or tokens.shape[1:]!=(3,16,1024) or positions.shape!=(len(tokens),16,58):raise ValueError('Aligned causal three-view tokens and measured state required')
        if method=='global_mean_control':tokens=tokens.mean(2,keepdim=True).expand(-1,-1,16,-1)
        elif method!='spatial_4x4':raise ValueError('Unknown fixed readout')
        z=self.visual(tokens.flatten(1,2))+self.position;q=self.state(positions.flatten(1))[:,None,:];a,_=self.attention(q,z,z,need_weights=False)
        return self.classifier(self.norm(q[:,0]+a[:,0]))
