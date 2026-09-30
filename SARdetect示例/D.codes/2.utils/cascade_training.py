"""Fine-tune installed models into isolated candidate runs; never overwrite deployment."""
import math,random
from project_io import ROOT,UTIL,PYTHON,YOLO_PYTHON,read,write,sha,manifest,checked
from cascade_runtime import run
from model_loader import load_model_module
from pathlib import Path
def train(key,data,out,args):
    out.mkdir(parents=True,exist_ok=True)
    if key in ['model-c-s','model-d-s']:return train_heads(key,data,out,args)
    row=manifest()['detectors'][key];raw=read(ROOT/row['config']);cp=checked(row)
    raw['run']=dict(id='fit',output_root=str(out));raw['model']['pretrained_detector']=str(cp)
    raw['runtime']['seed']=args.seed
    raw['data'].update(root=str(data/'annotations'),image_prefix=str(data/'images')+'/',data_yaml=str(data/'yolo.yaml'))
    for name,value in [('epochs',args.epochs),('batch_size',args.batch_size),('learning_rate',args.learning_rate)]:
        if value is not None:
            if value<=0:raise ValueError(f'{name} must be positive')
            raw['train'][name]=value
    if args.smoke:
        raw['data']['root']=str(out/'annotations')
        for split,limit in [('train',8),('val',4),('test',4)]:
            coco=read(data/f'annotations/{split}.json');coco['images']=coco['images'][:limit];ids={im['id'] for im in coco['images']};coco['annotations']=[a for a in coco['annotations'] if a['image_id'] in ids];write(out/f'annotations/{split}.json',coco)
        raw['train'].update(epochs=args.epochs or 1,workers=0,val_interval=1,log_interval=1,warmup_iters=1)
    raw['train']['milestones']=[max(1,raw['train']['epochs']-1)]
    if raw['backend']=='yolo':
        if args.smoke:raw['train'].update(nominal_batch_size=raw['train']['batch_size'],amp_initial_scale=128)
        from dataset_adapter import yolo_dataset
        raw['data']['data_yaml']=yolo_dataset(raw,out)
    from training_control import configure
    configure(raw,key)
    cfg=out/'resolved.json';write(cfg,raw)
    kind=raw['backend']
    if kind=='yolo':cmd=[YOLO_PYTHON,ROOT/'D.codes/1.model/model5s-YOLO11m/yolo_training.py','--config',cfg]+(['--smoke'] if args.smoke else [])
    else:
        script={'faster':'faster_backend.py','openmmlab':'openmmlab_backend.py','rtdetr':'rtdetr_backend.py'}[kind]
        cmd=[PYTHON,UTIL/script,'--config',cfg,'--run-id','fit']
        if kind!='faster':cmd+=['--mode','train']
        if kind=='rtdetr':cmd+=['--device','npu:0']
    run(cmd,out/'train.log')
    weights=Path(raw.get('training_control',{}).get('checkpoint_dir',out/'fit'))
    candidates=list(weights.rglob('best*.pth'))+list(weights.rglob('best.pt'))
    if not candidates:raise FileNotFoundError(f'No validation-selected checkpoint produced: {out}')
    write(out/'candidate.json',dict(status='candidate',model=key,parent_checkpoint=str(cp),checkpoints=[dict(path=str(p),sha256=sha(p)) for p in candidates],selection='native validation metric',automatic_deployment=False))

def train_heads(key,data,out,args):
    return load_model_module('model-d-s','train_refiner').train_heads(key,data,out,args)
