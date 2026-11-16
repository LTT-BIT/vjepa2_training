"""Current-frame right-hand/truss anchor; never follow left transport to basket.

Only target semantics are borrowed from new119, not its data or future-assisted
coordinates. Missing right anchor is explicitly invalid, never a left-only crop.
Visible fruit/context is evidence for crop extent, NOT physical success.
"""
import numpy as np


class CausalLocalROI:
    def __init__(self,joint_names,scale):
        names=list(joint_names);self.scale=np.maximum(np.asarray(scale,float),1e-4)
        if self.scale.shape!=(58,) or not np.isfinite(self.scale).all():raise ValueError('Finite train58 scale required')
        self.left=[i for i,n in enumerate(names) if n.startswith('left_')];self.right=[i for i,n in enumerate(names) if n.startswith('right_')]
        if len(self.left)!=29 or len(self.right)!=29:raise ValueError('Exact left29/right29 names required')

    def update(self,detections,q,width,height,rgb=None):
        q=np.asarray(q,float)
        if q.shape!=(58,) or not np.isfinite(q).all():raise ValueError('Finite measured58 required')
        best={}
        for row in detections:
            k=row.get('label');s=row.get('score');b=np.asarray(row.get('box'),float)
            if k not in (1,2,3) or not isinstance(s,(float,int)) or not np.isfinite(s) or not 0<=s<=1 or b.shape!=(4,) or not np.isfinite(b).all():continue
            if s<(.75 if k in (1,2) else .85) or not 0<=b[0]<b[2]<=width or not 0<=b[1]<b[3]<=height:continue
            area=np.prod(b[2:]-b[:2])
            if area>width*height*.13 or min(b[2:]-b[:2])<10:continue
            if k not in best or s>best[k][0]:best[k]=(s,b)
        if 1 not in best:return dict(valid=False,interaction_box=None,roles=[1])
        tight=best[1][1].copy();roles=[1]
        def gap(a,b):return float(np.linalg.norm(np.maximum(0,np.maximum(a[:2],b[:2])-np.minimum(a[2:],b[2:]))))
        def merged(a,b):return np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])]
        # Current RGB only. Nearby red may be fruit or background: this is an
        # experimental extent heuristic, never a held-object/success label.
        if rgb is not None:
            import cv2
            if rgb.shape!=(height,width,3) or rgb.dtype!=np.uint8:raise ValueError('Matching native current uint8 RGB required')
            hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
            mask=(((hsv[:,:,0]<13)|(hsv[:,:,0]>169))&(hsv[:,:,1]>90)&(hsv[:,:,2]>60)).astype(np.uint8)
            _,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
            candidates=[np.array([x,y,x+w,y+h],float) for x,y,w,h,area in stats[1:] if 8<=area<=3000 and min(w,h)>=3]
            attached=False
            for _ in range(4):
                retained=[];changed=False
                for b in candidates:
                    m=merged(tight,b)
                    if gap(tight,b)<=(15 if attached else 35) and max(m[2:]-m[:2])<=240:
                        tight=m;attached=True;changed=True
                    else:retained.append(b)
                candidates=retained
                if not changed:break
        if 3 in best:
            obj=best[3][1];m=merged(tight,obj)
            if gap(tight,obj)<30 and max(m[2:]-m[:2])<=240:tight=m
        # Left fingertips may join ONLY the anchored interaction region. A hand
        # moving alone at the basket cannot redirect the crop regardless of q.
        if 2 in best:
            left=best[2][1];m=merged(tight,left)
            if gap(tight,left)<15 and max(m[2:]-m[:2])<=240:tight=m;roles.append(2)
        tight+=np.array([-12,-18,12,18]);tight[:2]=np.maximum(tight[:2],0);tight[2:]=np.minimum(tight[2:],[width,height])
        return dict(valid=True,interaction_box=tight,roles=roles)
