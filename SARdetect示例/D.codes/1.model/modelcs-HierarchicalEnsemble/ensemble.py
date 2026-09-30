"""Portable hierarchical ensemble inference; every model and reference asset belongs to SARdetect."""
import os,sys,math,hashlib
from pathlib import Path
from functools import lru_cache
import cv2,numpy as np,torch
from torch import nn
from torchvision.models import convnext_tiny
from .hierarchy_model import Heads
from .level2_projection import expand_parent_probabilities
from .crop_processing import pixels as pixels160
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from dinov2.hub.backbones import dinov2_vitb14

class LeafModel(nn.Module):
 def __init__(self,tax,kind):
  super().__init__();self.kind=kind
  self.register_buffer('root_map',torch.tensor([[float(r['l1']==i) for i in range(len(tax['level1']))] for r in tax['leaves']]))
  self.register_buffer('parent_map',torch.tensor([[float(r['l2']==i) for i in range(len(tax['level2']))] for r in tax['leaves']]))
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1));self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
  if kind=='dino':self.backbone=dinov2_vitb14(pretrained=False,drop_path_rate=.1);channels=1536;dropout=.2
  else:self.features=convnext_tiny(weights=None,stochastic_depth_prob=.2).features;channels=768;dropout=.2
  self.head=nn.Sequential(nn.LayerNorm(channels),nn.Dropout(dropout),nn.Linear(channels,len(tax['leaves'])))
 def embedding(self,x):
  x=(x.repeat(1,3,1,1)-self.mean)/self.std
  if self.kind=='dino':
   f=self.backbone.forward_features(x);return torch.cat([f['x_norm_clstoken'],f['x_norm_patchtokens'].mean(1)],1)
  return self.features(x).mean((2,3))
 def forward(self,x):return self.head(self.embedding(x)).log_softmax(-1).exp()
 def retrieval(self,x,rotations=1,cls_weight=None):
  views=[nn.functional.normalize(self.head[0](self.embedding(torch.rot90(x,k*(4//rotations),(-2,-1)).contiguous())),dim=1) for k in range(rotations)]
  f=views[0] if rotations==1 else nn.functional.normalize(torch.stack(views).mean(0),dim=1)
  if cls_weight is None:return f
  return torch.cat([nn.functional.normalize(f[:,:768],dim=1)*cls_weight**.5,nn.functional.normalize(f[:,768:],dim=1)*(1-cls_weight)**.5],1)

def pixels224(arr,box):
 x,y,w,h=box;cx=x+w/2;cy=y+h/2;side=max(8,int(math.ceil(max(w,h)*1.1)))
 matrix=cv2.getRotationMatrix2D((cx,cy),0,1);matrix[0,2]+=side/2-cx;matrix[1,2]+=side/2-cy
 crop=cv2.warpAffine(arr,matrix,(side,side),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);lo,hi=np.percentile(crop,[1,99]);crop=np.clip((crop.astype(np.float32)-lo)/max(hi-lo,1),0,1)
 return cv2.resize(crop,(224,224),interpolation=cv2.INTER_LINEAR).astype(np.float16).astype(np.float32)

class ClassifierEnsemble:
 def __init__(self,root,tax,config,device='npu:0'):
  self.root=Path(root);self.tax=tax;self.cfg=config;self.device=device;self.checked=set()
 def asset(self,row):
  p=(self.root/row['path']).resolve();assert p.is_relative_to(self.root.resolve()),'Asset escapes SARdetect'
  if str(p) not in self.checked:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(2**20),b''):h.update(b)
   assert h.hexdigest()==row['sha256'],str(p);self.checked.add(str(p))
  return p
 def load(self,net,row):
  net.load_state_dict(torch.load(self.asset(row),map_location='cpu',weights_only=False)['model'],strict=True);return net.to(self.device).eval()
 @torch.no_grad()
 def predict(self,rows,coco,image_root,batch_size=64):
  if not rows:return np.empty((0,len(self.tax['leaves'])),np.float64)
  ims={v['id']:v['file_name'] for v in coco['images']}
  @lru_cache(maxsize=2)
  def arr(i):
   a=cv2.imread(str(Path(image_root)/ims[i]),0)
   if a is None:raise ValueError('Unreadable image '+ims[i])
   return a
  # Crop caching is bounded by this call's target count; no whole-dataset image cache.
  def batch(first,legacy=None):
   pp=rows[first:first+batch_size];x=np.stack([pixels160(arr(r['image_id']),r['bbox'],r['quad'],legacy) if legacy else pixels224(arr(r['image_id']),r['bbox']) for r in pp]);return torch.from_numpy(x[:,None]).to(self.device)
  legacy=[]
  for name,row in self.cfg['legacy'].items():
   net=self.load(Heads(self.tax,'classifier'),row);parts=[]
   for first in range(0,len(rows),batch_size):parts.append(net.log_probs(net(batch(first,row['mode'])))[2].exp().cpu().numpy())
   legacy.append(np.concatenate(parts));del net;torch.npu.empty_cache()
  current=np.mean(legacy,0);del legacy
  net=self.load(LeafModel(self.tax,'convnext'),self.cfg['convnext']);parts=[]
  for first in range(0,len(rows),batch_size):parts.append(net(batch(first)).cpu().numpy())
  conv=np.concatenate(parts);del net;torch.npu.empty_cache()
  net=self.load(LeafModel(self.tax,'dino'),self.cfg['dino']);parts=[];features=[]
  for first in range(0,len(rows),batch_size):
   x=batch(first);parts.append(net(x).cpu().numpy());features.append(net.retrieval(x,self.cfg.get('feature_rotations',1),self.cfg.get('feature_cls_weight')).cpu().numpy())
  dino=np.concatenate(parts);query=np.concatenate(features);del net;torch.npu.empty_cache()
  refs=torch.from_numpy(np.load(self.asset(self.cfg['reference_features']))).to(self.device);labels=np.load(self.asset(self.cfg['reference_labels']));parts=[]
  reference_depth=self.cfg.get('reference_depth',3)
  if reference_depth not in (2,3):raise ValueError('reference_depth must be 2 or 3')
  nclasses=len(self.tax['level2']) if reference_depth==2 else len(self.tax['leaves'])
  assert refs.shape==(len(labels),1536) and labels.min()>=0 and labels.max()<nclasses
  for first in range(0,len(rows),batch_size):
   values,indices=torch.topk(torch.from_numpy(query[first:first+batch_size]).to(self.device)@refs.T,min(100,len(labels)),dim=1);values=values.cpu().numpy();indices=indices.cpu().numpy();weights=np.exp((values[:,:20]-values[:,:1])/.1);weights=.99*weights/weights.sum(1,keepdims=True);p=np.full((len(values),nclasses),.01/nclasses,np.float64)
   for j in range(len(p)):np.add.at(p[j],labels[indices[j,:20]],weights[j])
   if reference_depth==2:p=expand_parent_probabilities(p,self.tax)
   parts.append(p)
  # Match frozen operation ordering and dtype, not just algebraic weights.
  neural=(current*(1/3)+dino*(1/3)+conv*(1/3));neural=np.maximum(neural,1e-12);neural/=neural.sum(1,keepdims=True)
  result=.75*neural+.25*np.concatenate(parts);assert np.isfinite(result).all() and np.max(abs(result.sum(1)-1))<1e-5
  return result
