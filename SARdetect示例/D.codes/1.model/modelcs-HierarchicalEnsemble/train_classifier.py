"""Warm-start all five deployed networks; rebuild retrieval from training targets only."""
import copy,random,time
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path
import numpy as np,cv2,torch,torch_npu
from torch import nn
from project_io import ROOT,read,write,sha,checked,manifest
from .hierarchy_model import Heads,crop,geometry_quad,class_metrics
from .ensemble import LeafModel,pixels224,ClassifierEnsemble
from .crop_processing import pixels
from training_control import EpochSession,checkpoint_dir

def train(key,data,out,args):
 torch.set_num_threads(4);torch.npu.set_device(0);cv2.setNumThreads(0)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.npu.manual_seed_all(args.seed)
 cfg=copy.deepcopy(manifest());tax=read(data/'taxonomy.json');assets=cfg['classifier']['assets'];out.mkdir(parents=True,exist_ok=True)
 cocos={s:read(data/f'annotations/{s}.json') for s in ['train','val']}
 if args.smoke:
  for coco in cocos.values():coco['annotations']=coco['annotations'][:12]
 assert all(c['annotations'] for c in cocos.values()),'Train and val must contain targets'
 rows={s:c['annotations'] for s,c in cocos.items()};ims={s:{im['id']:im['file_name'] for im in c['images']} for s,c in cocos.items()}
 mode=getattr(args,'classifier_mode','mixed')
 if mode not in ('mixed','level2'):raise ValueError('classifier_mode must be mixed or level2')
 if mode=='level2':
  depths=[r.get('mixed_label_depth',3) for r in rows['train']]
  if any(d>2 for d in depths):raise ValueError('level2 training requires training labels capped at depth 2')
  if not any(d==2 for d in depths):raise ValueError('level2 training requires at least one level-2 training target')
 from training_resources import ImageCache
 image_cache=ImageCache({(s,i):data/'images'/name for s,by in ims.items() for i,name in by.items()})
 def image(split,ident):
  return image_cache[(split,ident)]
 # Validation alignment always comes from the fixed deployed refiner.
 ref=Heads(tax,'refiner').to('npu:0');ref.load_state_dict(torch.load(checked(cfg['refiner']),map_location='cpu',weights_only=False)['model']);ref.eval();valquad={}
 with torch.no_grad():
  for r in rows['val']:
   x,rect=crop(image('val',r['image_id']),r['bbox']);g=ref(x[None].to('npu:0'))[0].cpu().numpy();valquad[r['id']]=geometry_quad(g,rect)
 del ref;torch.npu.empty_cache()
 names=list(assets['legacy'])+['convnext','dino'];names=names if args.branch=='all' else [args.branch]
 if mode=='level2' and 'dino' not in names:raise ValueError('level2 mode must train dino to rebuild the level-2 retrieval index')
 defaults=read(ROOT/'D.codes/1.model/modelcs-HierarchicalEnsemble/config.json')['train']
 epochs=args.epochs or (1 if args.smoke else defaults['epochs']);bs=args.batch_size or defaults['batch_size'];lr=args.learning_rate or defaults['learning_rate']
 if min(epochs,bs,lr)<=0:raise ValueError('Training parameters must be positive')
 def tensor(split,rr,name,augment=False):
  xx=[]
  for r in rr:
   arr=image(split,r['image_id'])
   if name in assets['legacy']:
    q=r['quad'] if split=='train' else valquad[r['id']];x=pixels(arr,r['bbox'],q,assets['legacy'][name]['mode'])
   else:x=pixels224(arr,r['bbox'])
   if augment:
    if random.random()<.5:x=x[:,::-1]
    if random.random()<.5:x=x[::-1]
   xx.append(np.ascontiguousarray(x))
  return torch.from_numpy(np.stack(xx)[:,None]).to('npu:0')
 def probs(net,x,name):
  if name in assets['legacy']:return net.log_probs(net(x))
  p=net(x).clamp_min(1e-12);return (p@net.root_map).clamp_min(1e-12).log(),(p@net.parent_map).clamp_min(1e-12).log(),p.log()
 history_all={}
 for name in names:
  dest=out/name;dest.mkdir(exist_ok=True);legacy=name in assets['legacy'];row=assets['legacy'][name] if legacy else assets[name];cp=ROOT/row['path'];assert sha(cp)==row['sha256']
  net=(Heads(tax,'classifier') if legacy else LeafModel(tax,name)).to('npu:0');net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'],strict=True)
  opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.05);best=-1;history=[];start=time.monotonic();processed=0
  write(dest/'config.json',dict(branch=name,seed=args.seed,epochs=epochs,batch_size=bs,learning_rate=lr,parent=str(cp),parent_sha256=sha(cp),selection='val per-image hierarchical score',partial_labels='mask unknown deeper targets; no invented fine labels',dataset=str(data),classifier_mode=mode))
  session=EpochSession(key,name,net,opt,epochs,dict(model_name=key,dataset=str(data),train_targets=len(rows['train']),val_targets=len(rows['val']),batch_size=bs,learning_rate=lr,seed=args.seed))
  best=session.best;history=session.history;weights=session.path
  for epoch in range(session.start_epoch,epochs):
   net.train();order=np.random.permutation(len(rows['train']));loss_total=0
   for first in range(0,len(order),bs):
    rr=[rows['train'][i] for i in order[first:first+bs]];x=tensor('train',rr,name,True);pp=probs(net,x,name);loss=0
    for level,w in enumerate([1,.8,1]):
     valid=[j for j,r in enumerate(rr) if r.get('mixed_label_depth',3)>=level+1 and r[f'l{level+1}']>=0]
     if valid:
      y=torch.tensor([rr[j][f'l{level+1}'] for j in valid],device='npu:0');loss=loss+w*nn.functional.nll_loss(pp[level][valid],y)
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();loss_total+=float(loss.detach().cpu())*len(rr);processed+=len(rr)
    session.progress(epoch,first//bs+1,(len(order)+bs-1)//bs,float(loss.detach().cpu()))
   if not session.validate(epoch):
    session.finish(epoch,loss_total/len(rows['train']),{},None,dict(taxonomy=tax));write(dest/'history.json',session.history);continue
   net.eval();pred=[]
   with torch.no_grad():
    for first in range(0,len(rows['val']),bs):pred.extend(probs(net,tensor('val',rows['val'][first:first+bs],name),name)[2].exp().cpu().numpy())
   metric=class_metrics(np.asarray(pred),SimpleNamespace(rows=rows['val']),tax,0,prior=args.depth_prior,output_depth=2 if mode=='level2' else None);score=metric['mixed_score']
   session.finish(epoch,loss_total/len(rows['train']),metric,score,dict(taxonomy=tax));best=session.best;write(dest/'history.json',session.history)

  net.load_state_dict(torch.load(weights/'best.pth',map_location='cpu',weights_only=False)['model'],strict=True);elapsed=session.total_seconds;del session,net,opt;torch.npu.empty_cache()
  row.update(path=str((weights/'best.pth').relative_to(ROOT)),sha256=sha(weights/'best.pth'));row.pop('bytes',None)
  history_all[name]=dict(epochs=epochs,processed_targets=epochs*len(rows['train']),attempt_processed_targets=processed,seconds=elapsed,best=best,reload_verified=True)
  write(out/'training_summary.json',history_all)
 # Refresh retrieval whenever the DINO encoder changes; never reuse incompatible features.
 if 'dino' in names:
  rr=[r for r in rows['train'] if r.get('mixed_label_depth',3)>=2 and r['l2']>=0] if mode=='level2' else [r for r in rows['train'] if r.get('mixed_label_depth',3)==3 and r['l3']>=0]
  if not rr:raise ValueError('Retrieval requires at least one '+('level-2' if mode=='level2' else 'fine-labeled')+' training target')
  net=LeafModel(tax,'dino').to('npu:0');net.load_state_dict(torch.load(ROOT/assets['dino']['path'],map_location='cpu',weights_only=False)['model']);net.eval();feats=[]
  with torch.no_grad():
   for first in range(0,len(rr),bs):feats.extend(net.retrieval(tensor('train',rr[first:first+bs],'dino')).cpu().numpy())
  del net;torch.npu.empty_cache();np.save(checkpoint_dir(key)/'reference_features.npy',np.asarray(feats,np.float32));np.save(checkpoint_dir(key)/'reference_labels.npy',np.asarray([r['l2' if mode=='level2' else 'l3'] for r in rr],np.int64))
  for name in ['reference_features','reference_labels']:
   path=checkpoint_dir(key)/(name+'.npy');assets[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
  write(out/'reference_provenance.json',dict(source='training split only',dataset=str(data),targets=len(rr),annotation_ids=[r['id'] for r in rr],encoder_sha256=assets['dino']['sha256'],validation_or_test_targets=0,label_depth=2 if mode=='level2' else 3))
 if mode=='level2':
  assets['reference_depth']=2
  cfg['classifier']['output_depth']=2
 cfg['classifier']['depth_prior']=list(args.depth_prior);cfg['classifier']['selected_phase']='candidate-'+out.parent.name
 write(out/'candidate_manifest.json',cfg)
 write(out/'candidate.json',dict(status='candidate',model=key,manifest=str(out/'candidate_manifest.json'),sha256=sha(out/'candidate_manifest.json'),trained_branches=names,automatic_deployment=False,training=history_all))
