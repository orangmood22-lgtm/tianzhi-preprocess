"""Installed SIVED train/test/infer entrypoints with per-image detection scoring."""
import argparse,os,sys,json,time,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];UTIL=ROOT/'D.codes/2.utils'
from training_resources import interpreter
PY=interpreter();YP=interpreter(True)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def normalize(value):
 value=str(value).removeprefix('model-')
 if value not in ['1','2','3','4','5']:raise argparse.ArgumentTypeError('SIVED models: model-1..model-5')
 return 'model-'+value
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','test','infer','pipeline'])
 p.add_argument('--model',type=normalize,default='model-1');p.add_argument('--models',nargs='+',type=normalize)
 p.add_argument('--dataset',default='SIVED_OBB');p.add_argument('--config',default='00');p.add_argument('--device',default='npu:0');p.add_argument('--split',choices=['val','test'],default='test')
 p.add_argument('--devices',help='Comma-separated physical NPUs for DDQ/DINO training');p.add_argument('--master-port',type=int,default=29780)
 p.add_argument('--seed',type=int,default=3929);p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--learning-rate',type=float);p.add_argument('--smoke',action='store_true')
 p.add_argument('--finetune',action='store_true',help='Warm-start existing detector/OBB with short warmup; learning rate supplied by workflow')
 p.add_argument('--checkpoint',type=Path);p.add_argument('--score-threshold',type=float);p.add_argument('--samples',type=int);p.add_argument('--run-id');p.add_argument('--time-code');p.add_argument('--output',type=Path);p.add_argument('--dry-run',action='store_true');p.add_argument('--dump-predictions',action='store_true')
 p.add_argument('--obb-epochs',type=int,default=12);p.add_argument('--obb-batch-size',type=int,default=64);p.add_argument('--obb-learning-rate',type=float,default=.0001);p.add_argument('--obb-checkpoint',type=Path);p.add_argument('--hbb-only',action='store_true');p.add_argument('--select-threshold',action='store_true')
 a=p.parse_args()
 if min(a.obb_epochs,a.obb_batch_size,a.obb_learning_rate)<=0:p.error('OBB training parameters must be positive')
 if a.select_threshold and (a.action=='train' or a.split!='val'):p.error('Threshold selection is restricted to val')
 device=(a.devices or a.device).removeprefix('npu:');devices=device.split(',')
 if any(not d.isdigit() for d in devices) or len(set(devices))!=len(devices):p.error('Use distinct numeric physical NPUs')
 if len(devices)>1 and a.action!='train':p.error('Multiple NPUs are supported for training only')
 if a.config not in ['00','config-00']:p.error('Registered SIVED refresh uses config-00')
 if a.samples is not None and a.samples<1:p.error('--samples must be positive')
 if a.score_threshold is not None and not 0<=a.score_threshold<=1:p.error('Threshold outside [0,1]')
 os.environ['ASCEND_RT_VISIBLE_DEVICES']=device
 models=a.models or [a.model];definition=read(ROOT/'B.models/sived_manifest.json');source=ROOT/'A.dataset'/a.dataset
 if a.time_code and (not a.time_code.isdigit() or len(a.time_code) not in [8,14]):p.error('--time-code must be YYYYMMDD or YYYYMMDDHHMMSS')
 runid=a.run_id or 'sived-'+a.action+'-'+((a.time_code+'-') if a.time_code else '')+str(time.time_ns())
 if Path(runid).name!=runid or runid in ['.','..']:p.error('--run-id must be a directory name; use --output for a custom path')
 out=(a.output or ROOT/'C.results'/('1.train' if a.action=='train' else '3.infer' if a.action=='infer' else '2.test')/runid).resolve()
 if out.exists() and not os.environ.get('SAR_TRAIN_CONTEXT'):raise FileExistsError(out)
 write(out/'request.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},pid=os.getpid(),manifest_sha256=sha(ROOT/'B.models/sived_manifest.json')))
 if a.dry_run:write(out/'completed.json',dict(status='dry-run',models=models));return
 from yolo_data import write_coco,read_split,split_overlap_audit
 assert not any(split_overlap_audit(source).values()),'Split overlap'
 results={}
 for key in models:
  from training_control import already_completed
  if a.action=='train' and already_completed(out/key):results[key]=read(out/key/'candidate.json');continue
  dest=out/key;row=definition['models'][key];cp=(a.checkpoint or ROOT/row['checkpoint']).resolve(strict=True)
  if not a.checkpoint:assert sha(cp)==row['sha256']
  obb=row.get('obb');selected_threshold=row['threshold']
  if cp.suffix=='.json':
   bundle=read(cp);obb=bundle.get('obb');selected_threshold=bundle.get('threshold',selected_threshold);entry=bundle['checkpoints'][0];cp=(ROOT/entry['path']).resolve(strict=True);assert sha(cp)==entry['sha256']
  if a.obb_checkpoint:obb=dict(checkpoint=str(a.obb_checkpoint.resolve(strict=True)),sha256=sha(a.obb_checkpoint))
  raw=read(ROOT/row['config']);kind=raw['backend'];raw['model']['pretrained_detector']=str(cp)
  if len(devices)>1 and kind!='openmmlab':p.error('Distributed entrypoint supports DDQ/DINO')
  raw['runtime'].update(world_size=len(devices),physical_devices=device,seed=a.seed)
  raw['run']=dict(id='fit' if a.action=='train' else 'evaluate',output_root=str(dest/'raw'));raw['data'].update(root=str(dest/'annotations'),image_prefix=str(source/'images')+'/')
  for split in ['train','val','test']:
   f=dest/f'annotations/{split}.json';f.parent.mkdir(parents=True,exist_ok=True);write_coco(source,split,f)
   limit=(8 if split=='train' else 4) if a.smoke else ((a.samples or 12) if a.action=='infer' and split==a.split else a.samples if a.action!='train' and split==a.split else None)
   if limit:
    coco=read(f);coco['images']=coco['images'][:limit];ids={im['id'] for im in coco['images']};coco['annotations']=[x for x in coco['annotations'] if x['image_id'] in ids];write(f,coco)
  if a.action=='train' and not a.hbb_only:
   for split in ['train','val']:
    if any('quad' not in r for r in read(dest/f'annotations/{split}.json')['annotations']):raise ValueError('OBB retraining requires four-point labels; use SIVED_OBB or an OBB competition dataset')
  if a.action=='train':
   for field,value in [('epochs',a.epochs),('batch_size',a.batch_size),('learning_rate',a.learning_rate)]:
    if value is not None:
     if value<=0:raise ValueError(field+' must be positive')
     raw['train'][field]=value
   if a.smoke:raw['train'].update(epochs=a.epochs or 1,workers=0,warmup_iters=1,log_interval=1)
   raw['train']['milestones']=[max(1,raw['train']['epochs']-2),max(1,raw['train']['epochs']-1)]
   if a.finetune:
    from finetune_settings import configure_finetune
    configure_finetune(raw,len(read(dest/'annotations/train.json')['images']))
   if kind=='yolo':
    if a.smoke:raw['train'].update(nominal_batch_size=raw['train']['batch_size'],amp_initial_scale=128)
    from dataset_adapter import yolo_dataset
    raw['data']['data_yaml']=yolo_dataset(raw,dest)
  if a.action=='train':
   from training_control import configure
   configure(raw,key)
  config=dest/'resolved.json';write(config,raw)
  if kind=='openmmlab' and a.action=='train' and len(devices)>1:
   raw['train']['per_device_batch']=raw['train']['batch_size'];write(config,raw)
   cache=dest/'cache'/device.replace(',','-');cache.mkdir(parents=True,exist_ok=True);os.environ['ASCEND_CACHE_PATH']=str(cache)
   cmd=[PY,'-m','torch.distributed.run',f'--nproc_per_node={len(devices)}',f'--master_port={a.master_port}',UTIL/'sived_ddp_backend.py','--config',config,'--run-id',raw['run']['id'],'--mode','train']
  elif kind=='yolo' and a.action=='train':cmd=[YP,UTIL/'sived_yolo_train.py','--config',config]+(['--smoke'] if a.smoke else [])
  else:
   script={'faster':'faster_backend.py' if a.action=='train' else 'faster_evaluate.py','openmmlab':'openmmlab_backend.py','rtdetr':'rtdetr_backend.py','yolo':'yolo_backend.py'}[kind]
   cmd=[YP if kind=='yolo' else PY,UTIL/script,'--config',config,'--run-id',raw['run']['id']]
   if kind!='faster':cmd+=['--mode','train' if a.action=='train' else 'test']
   if kind in ['rtdetr','yolo']:cmd+=['--device','npu:0']
   if a.action!='train':
    cmd+=['--checkpoint',cp,'--split',a.split,'--dump-predictions']
    if kind=='faster':cmd+=['--output-root',dest/'raw']
  write(dest/'command.json',dict(command=list(map(str,cmd)),checkpoint=str(cp),sha256=sha(cp)))
  from training_control import tee
  detector_done=dest/'detector_candidate.json'
  if a.action!='train' or not detector_done.exists():tee(cmd,dest/'console.log')
  if a.action=='train':
   weights=list(Path(raw.get('training_control',{}).get('checkpoint_dir',dest/'raw')).rglob('best*.pth'))+list(Path(raw.get('training_control',{}).get('checkpoint_dir',dest/'raw')).rglob('best.pt'));weights=[w for w in weights if 'obb' not in w.parts];assert weights,'No best checkpoint'
   result=read(detector_done) if detector_done.exists() else dict(status='candidate',model=key,automatic_deployment=False,checkpoints=[dict(path=str(w),sha256=sha(w)) for w in weights]);write(detector_done,result)
   if not a.hbb_only:
    from sived_obb import train as train_obb
    initial=(ROOT/obb['checkpoint']) if obb else ROOT/'B.models/model-d-s/SIVED_ATR_v3_effb0_obb_stage14.pth'
    if obb:assert sha(initial)==obb['sha256']
    result['obb']=train_obb(key,dest/'annotations',source/'images',dest,a,initial);result.update(box_type='OBB',architecture='HBB detector + separately trained SIVED OBB regression',threshold=selected_threshold)
   results[key]=result;write(dest/'candidate.json',result);continue
  import torch,torch_npu
  from result_artifacts import _coco_predictions
  from sived_detection_metrics import metric,coco_metrics
  coco=read(dest/f'annotations/{a.split}.json');pp=_coco_predictions(dest/'raw/evaluate',coco);write(dest/'predictions.json',pp)
  threshold=selected_threshold if a.score_threshold is None else a.score_threshold
  if obb and not a.hbb_only:
   from sived_obb import refine,metric as obb_metric,choose_threshold
   ocp=(ROOT/obb['checkpoint']).resolve(strict=True);assert sha(ocp)==obb['sha256']
   write(dest/'hbb_predictions.json',pp);pp=refine(coco,pp,source/'images',ocp,floor=min(.05,threshold));write(dest/'predictions.json',pp);metric=obb_metric
   if a.select_threshold:
    selected,table=choose_threshold(coco,pp);threshold=selected['threshold'];write(dest/'threshold_selection.json',dict(selected=threshold,split='val',trials=table))
  if os.environ.get('SARDETECT_UNLABELED')=='1':
   result=dict(images=len(coco['images']),detections=sum(q['score']>=threshold for q in pp),labels_available=False)
  else:
   result=metric(coco,pp,threshold)
   if not obb or a.hbb_only:result['coco']=coco_metrics(coco,pp)
   result['checkpoint_sha256']=sha(cp);result['obb_sha256']=obb['sha256'] if obb and not a.hbb_only else None;write(dest/'metrics.json',result)
  results[key]={k:v for k,v in result.items() if k not in ['per_image','by_size']}
  if a.action=='infer':
   from PIL import Image,ImageDraw
   for im in coco['images']:
    image=Image.open(source/'images'/im['file_name']).convert('RGB');draw=ImageDraw.Draw(image)
    for q in pp:
     if q['image_id']==im['id'] and q['score']>=threshold:
      x,y,w,h=q['bbox']
      if 'quad' in q:
       points=[tuple(pt) for pt in q['quad']];draw.line(points+[points[0]],fill='red',width=2)
      else:draw.rectangle((x,y,x+w,y+h),outline='red',width=2)
      draw.text((x,y),f"vehicle {q['score']:.2f}",fill='yellow')
    image.save(dest/f"sample_{im['id']}.png")
 write(out/'summary.json',results);write(out/'completed.json',dict(status='passed',models=models,finished=time.time()));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
