"""Hierarchical classifier and shared learned OBB refiner for v3."""
import argparse,os,json,random,math,time,hashlib,functools
from pathlib import Path
import cv2,numpy as np
import torch,torch_npu
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import efficientnet_b0
from competition_metrics import actions,decode,reward,poly_iou

def read(p):return json.loads(Path(p).read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))
@functools.lru_cache(maxsize=64)
def image(path):return cv2.imread(path,0)
def crop(arr,box,size=160,context=1.25):
 x,y,w,h=box;cx=x+w/2;cy=y+h/2;w=max(w*context,8);h=max(h*context,8)
 x0=max(0,int(math.floor(cx-w/2)));y0=max(0,int(math.floor(cy-h/2)));x1=min(arr.shape[1],int(math.ceil(cx+w/2)));y1=min(arr.shape[0],int(math.ceil(cy+h/2)))
 assert x1>x0 and y1>y0
 a=arr[y0:y1,x0:x1];lo,hi=np.percentile(a,[1,99]);a=np.clip((a.astype(np.float32)-lo)/max(hi-lo,1),0,1);a=cv2.resize(a,(size,size))
 return torch.from_numpy(a[None]),np.array([x0,y0,x1-x0,y1-y0],np.float32)
def target_geometry(quad,rect):
 q=np.asarray(quad,np.float32); (cx,cy),(w,h),ang=cv2.minAreaRect(q)
 if h>w:w,h=h,w;ang+=90
 x,y,cw,ch=rect;s=math.sqrt(cw*ch);theta=math.radians(ang)
 return np.asarray([(cx-x)/cw-.5,(cy-y)/ch-.5,math.log(max(w,2)/s),math.log(max(h,2)/s),math.cos(2*theta),math.sin(2*theta)],np.float32)
def geometry_quad(value,rect):
 x,y,cw,ch=rect;s=math.sqrt(cw*ch);dx,dy,lw,lh,c,sn=value
 cx=x+(.5+np.clip(dx,-.75,.75))*cw;cy=y+(.5+np.clip(dy,-.75,.75))*ch
 w=np.exp(np.clip(lw,-3,1))*s;h=np.exp(np.clip(lh,-3,1))*s;angle=math.degrees(.5*math.atan2(sn,c))
 return cv2.boxPoints(((float(cx),float(cy)),(float(w),float(h)),float(angle))).tolist()
class Heads(nn.Module):
 def __init__(self,tax,kind,pretrained=False):
  super().__init__();net=efficientnet_b0(weights=None)
  if pretrained:raise ValueError("Load an explicit registered checkpoint")
  conv=net.features[0][0];gray=nn.Conv2d(1,conv.out_channels,conv.kernel_size,conv.stride,conv.padding,bias=False)
  with torch.no_grad():gray.weight.copy_(conv.weight.sum(1,keepdim=True))
  net.features[0][0]=gray;self.features=net.features;self.kind=kind;self.tax=tax
  if kind=='classifier':self.h1=nn.Linear(1280,len(tax['level1']));self.h2=nn.Linear(1280,len(tax['level2']));self.h3=nn.Linear(1280,len(tax['leaves']))
  else:
   self.reg=nn.Linear(1280,6);nn.init.zeros_(self.reg.weight)
   with torch.no_grad():self.reg.bias.copy_(torch.tensor([0,0,-.2,-.8,1,0.]))
 def forward(self,x):
  f=self.features(x).mean((2,3))
  if self.kind=='refiner':return self.reg(f)
  return self.h1(f),self.h2(f),self.h3(f)
 def log_probs(self,out):
  a,b,c=out;pa=a.log_softmax(-1);pb=torch.zeros_like(b);pc=torch.zeros_like(c)
  for i in range(len(self.tax['level1'])):
   ids=[j for j,r in enumerate(self.tax['level2']) if r['parent']==i];pb[:,ids]=b[:,ids].log_softmax(-1)+pa[:,i,None]
  for i in range(len(self.tax['level2'])):
   ids=[r['l3'] for r in self.tax['leaves'] if r['l2']==i];pc[:,ids]=c[:,ids].log_softmax(-1)+pb[:,i,None]
  return pa,pb,pc
def class_metrics(probs,ds,tax,margin=0,prior=(.1,.3,.6),output_depth=None):
 aa=actions(tax);pred,_=decode(probs,tax,margin,prior,output_depth);by={};byfull={};correct=[]
 for k,a in enumerate(ds.rows):
  truth=(a['l1'],a['l2'],a['l3']);by.setdefault(a['image_id'],[]).append(reward(aa[pred[k]],truth,a['mixed_label_depth']))
  if a['l3']>=0:
   byfull.setdefault(a['image_id'],[]).append(reward(aa[pred[k]],truth,3));correct.append(int(probs[k].argmax()==a['l3']))
 return dict(mixed_score=float(np.mean([np.mean(v) for v in by.values()])),full_depth_score=float(np.mean([np.mean(v) for v in byfull.values()])) if byfull else None,leaf_top1=float(np.mean(correct)) if correct else None,known_fine_targets=len(correct),output_depth_counts={str(d):int(sum(aa[i][0]==d for i in pred)) for d in (1,2,3)})
