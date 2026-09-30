"""Train/test/infer the retained ResCNN classifier and residual HBB refiner."""
import argparse,os,json,time,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from project_io import read,write,sha
from training_control import EpochSession
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','test','infer','pipeline']);p.add_argument('--models',nargs='+');p.add_argument('--model');p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--device',default='npu:0');p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--config',default='00');p.add_argument('--seed',type=int,default=3926);p.add_argument('--samples',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--run-id');p.add_argument('--output',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 if a.config not in ['00','config-00']:p.error('Registered components use config-00')
 a.epochs=a.epochs or (1 if a.smoke else 12)
 dev=a.device.removeprefix('npu:');assert dev.isdigit();os.environ['ASCEND_RT_VISIBLE_DEVICES']=dev
 import torch,torch_npu,numpy as np
 from PIL import Image
 from functools import lru_cache
 from pipeline_components import ATRClassifier,ResidualPixelRefiner,image_tensor,square_crop,encode_box,decode_box,xywh_to_xyxy,box_iou
 torch.npu.set_device(0);torch.set_num_threads(4);torch.manual_seed(a.seed);torch.npu.manual_seed_all(a.seed);random.seed(a.seed)
 if min(a.batch_size,a.epochs,a.learning_rate)<=0:raise ValueError('Positive training parameters required')
 keys=a.models or [a.model];assert all(k in ['model-c','model-d'] for k in keys)
 out=a.output or ROOT/'C.results'/('1.train' if a.action=='train' else '3.infer' if a.action=='infer' else '2.test')/(a.run_id or 'legacy-'+str(time.time_ns()));out.mkdir(parents=True,exist_ok=bool(os.environ.get('SAR_TRAIN_CONTEXT')))
 write(out/'request.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},pid=os.getpid()))
 if a.dry_run:write(out/'completed.json',{'status':'dry-run'});return
 source=ROOT/'A.dataset'/a.dataset;catalog=read(ROOT/'D.codes/1.model/catalog.json');cocos={}
 for s in ['train','val',a.split]:
  if (source/f'annotations/{s}.json').exists():cocos[s]=read(source/f'annotations/{s}.json')
  else:
   from yolo_data import write_coco
   tmp=out/f'{s}.json';write_coco(source,s,tmp);cocos[s]=read(tmp)
  if a.smoke or a.samples:
   c=cocos[s];c['images']=c['images'][:a.samples or 4];ids={im['id'] for im in c['images']};c['annotations']=[r for r in c['annotations'] if r['image_id'] in ids][:32 if a.smoke else 10**12]
 @lru_cache(maxsize=4)
 def image(s,i):
  name=next(im['file_name'] for im in cocos[s]['images'] if im['id']==i)
  with Image.open(source/'images'/name) as im:return im.convert('L').copy()
 summary={}
 for key in keys:
  from training_control import already_completed
  if a.action=='train' and already_completed(out/key):summary[key]={'status':'candidate','resumed_completed_model':True};continue
  dest=out/key;dest.mkdir(exist_ok=True);cfg=catalog[key];cp=a.checkpoint or ROOT/cfg['checkpoint'];state=torch.load(cp,map_location='cpu',weights_only=False)
  if not a.checkpoint:assert sha(cp)==cfg['sha256']
  classifier=key=='model-c';net=(ATRClassifier(40) if classifier else ResidualPixelRefiner()).to('npu:0');net.load_state_dict(state['model'],strict=True)
  tax=read(ROOT/'D.codes/1.model/modelcs-HierarchicalEnsemble/taxonomy.json');class_ids=state['metadata']['classes'] if classifier else None
  def prepare(s,rr,jitter=False):
   xx=[];gg=[];yy=[];rects=[]
   for r in rr:
    box=xywh_to_xyxy(r['bbox']);proposal=box[:]
    if not classifier and jitter:
     w,h=box[2]-box[0],box[3]-box[1];proposal=[box[0]+.03*w,box[1]-.02*h,box[2]-.03*w,box[3]+.02*h]
    rect=square_crop(proposal,2 if classifier else 1.75);xx.append(image_tensor(image(s,r['image_id']).crop(tuple(rect)),128));gg.append(encode_box(proposal,rect));rects.append(rect)
    if classifier:
     if r.get('mixed_label_depth',3)<3 or r.get('l3',-1)<0:yy.append(-1)
     else:yy.append(class_ids[tax['leaves'][r['l3']]['level3']])
    else:yy.append(encode_box(box,rect))
   return torch.stack(xx).to('npu:0'),torch.tensor(gg,device='npu:0'),torch.tensor(yy,device='npu:0'),rects
  def evaluate(s):
   net.eval();pred=[];vals=[]
   with torch.no_grad():
    for first in range(0,len(cocos[s]['annotations']),a.batch_size):
     rr=cocos[s]['annotations'][first:first+a.batch_size];x,g,y,rects=prepare(s,rr,True)
     if classifier:
      z=net(x).argmax(1).cpu().tolist();vals.extend([float(v==int(t)) for v,t in zip(z,y.cpu()) if int(t)>=0]);pred.extend([dict(image_id=r['image_id'],bbox=r['bbox'],class_id=v) for r,v in zip(rr,z)])
     else:
      z=net(x,g)[0].cpu().tolist()
      for r,v,rect in zip(rr,z,rects):
       box=decode_box(v,rect);vals.append(box_iou(box,xywh_to_xyxy(r['bbox'])));pred.append(dict(image_id=r['image_id'],bbox_xyxy=box))
   if not vals:raise ValueError('No evaluable targets for retained component')
   return dict(**({'leaf_top1':float(np.mean(vals))} if classifier else {'mean_iou':float(np.mean(vals))}),targets=len(vals),protocol='GT proposal retained-component diagnostic; not competition score'),pred
  if a.action=='train':
   opt=torch.optim.AdamW(net.parameters(),lr=a.learning_rate);history=[];best=-1;start=time.monotonic()
   session=EpochSession(key,'component',net,opt,a.epochs,dict(model_name=key,dataset=str(source),train_targets=len(cocos['train']['annotations']),val_targets=len(cocos['val']['annotations']),batch_size=a.batch_size,learning_rate=a.learning_rate,seed=a.seed))
   best=session.best;weights=session.path
   for epoch in range(session.start_epoch,a.epochs):
    net.train();rows=list(cocos['train']['annotations']);random.shuffle(rows);losses=[]
    for first in range(0,len(rows),a.batch_size):
     rr=rows[first:first+a.batch_size];x,g,y,_=prepare('train',rr,True)
     if classifier:
      if not (y>=0).any():continue
      loss=torch.nn.functional.cross_entropy(net(x)[y>=0],y[y>=0])
     else:loss=torch.nn.functional.smooth_l1_loss(net(x,g)[0],y)
     assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
     session.progress(epoch,first//a.batch_size+1,(len(rows)+a.batch_size-1)//a.batch_size,losses[-1])
    if not losses:raise ValueError('No labeled training targets')
    metric,_=evaluate('val') if session.validate(epoch) else ({},None);score=metric.get('leaf_top1' if classifier else 'mean_iou')
    session.finish(epoch,float(np.mean(losses)),metric,score,dict(metadata=state['metadata']));write(dest/'history.json',session.history)
   net.load_state_dict(torch.load(weights/'best.pth',map_location='cpu',weights_only=False)['model']);write(dest/'candidate.json',dict(status='candidate',model=key,checkpoints=[{'path':str(weights/'best.pth'),'sha256':sha(weights/'best.pth')}],automatic_deployment=False,reload_verified=True,seconds=time.monotonic()-start));summary[key]={'status':'candidate'}
  else:
   metric,pred=evaluate(a.split);write(dest/'metrics.json',metric);write(dest/'predictions.json',pred);write(dest/'checkpoint.json',dict(path=str(cp),sha256=sha(cp)));summary[key]=metric
  del net;torch.npu.empty_cache()
 write(out/'summary.json',summary);write(out/'completed.json',dict(status='passed',models=keys));print(out)
if __name__=='__main__':main()
