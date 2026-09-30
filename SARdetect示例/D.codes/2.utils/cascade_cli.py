"""Train/test/infer entrypoint for the registered synthetic-v3 models."""
import argparse,os,sys,time
from pathlib import Path
import cv2,numpy as np,json
from project_io import ROOT,read,write,sha,manifest
from cascade_runtime import dataset,detect,Pipeline
from model_loader import load_model_module
def normalize(v):
    if v in ['1','2','3','4','5']:v=f'model-{v}-s'
    if v in [f'{i}-s' for i in range(1,6)]:v='model-'+v
    if v not in [f'model-{i}-s' for i in range(1,6)]+['model-c-s','model-d-s']:raise argparse.ArgumentTypeError('Use model-1-s..model-5-s, model-c-s or model-d-s')
    return v
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['train','test','infer','pipeline'])
    ap.add_argument('--model',type=normalize);ap.add_argument('--models',nargs='+',type=normalize)
    ap.add_argument('--v3',action='store_true');ap.add_argument('--dataset');ap.add_argument('--split',choices=['val','test'],default='test')
    ap.add_argument('--device',default='npu:0');ap.add_argument('--devices',help='One physical NPU; multiple cards are not silently ignored')
    ap.add_argument('--run-id');ap.add_argument('--output',type=Path);ap.add_argument('--samples',type=int)
    ap.add_argument('--config',default='00');ap.add_argument('--dump-predictions',action='store_true')
    ap.add_argument('--classifier',choices=['model-c-s','phase19-hierarchy-ensemble'],default='model-c-s')
    ap.add_argument('--second-stage',choices=['model-d-s'],default='model-d-s')
    ap.add_argument('--epochs',type=int);ap.add_argument('--batch-size',type=int);ap.add_argument('--learning-rate',type=float)
    ap.add_argument('--seed',type=int,default=3923);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--branch',choices=['baseline','aligned','aligned3924','convnext','dino','all'],default='all')
    ap.add_argument('--hbb-predictions',type=Path,help='Replay HBB export for adapter verification; one model only')
    ap.add_argument('--depth-prior',nargs=3,type=float,metavar=('P1','P2','P3'),help='Ground-truth depth prior for the decision gate; non-negative, sum=1; default from manifest (0.1 0.3 0.6)')
    ap.add_argument('--classifier-mode',choices=['mixed','level2'],default='mixed',help='Training mode; level2 expects capped training labels and preserves original validation labels')
    ap.add_argument('--output-depth',type=int,choices=[1,2,3],help='Force output label depth during evaluation/inference')
    ap.add_argument('--checkpoint',type=Path,help='Detector/refiner weights or classifier candidate_manifest.json')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();a.device=(a.devices or a.device).removeprefix('npu:')
    if a.classifier_mode!='mixed' and (a.action!='train' or (a.models or [a.model or 'model-1-s'])!=['model-c-s']):ap.error('--classifier-mode level2 is only for model-c-s training')
    if a.output_depth is not None and a.action=='train':ap.error('--output-depth is for evaluation/inference')
    from competition_metrics import validate_depth_prior
    try:a.depth_prior=list(validate_depth_prior(a.depth_prior if a.depth_prior is not None else manifest()['classifier']['depth_prior']))
    except ValueError as e:ap.error(str(e))
    if not a.device.isdigit():ap.error('Select one NPU, e.g. --device npu:0')
    if a.config not in ['00','config-00']:ap.error('v3 provides the frozen config-00')
    if a.samples is not None and a.samples<1:ap.error('--samples must be positive')
    os.environ['ASCEND_RT_VISIBLE_DEVICES']=a.device
    keys=a.models or [a.model or 'model-1-s']
    if a.action=='pipeline' and not a.models and not a.model:keys=list(manifest()['detectors'])
    if a.hbb_predictions and len(keys)!=1:ap.error('HBB replay requires one model')
    data=dataset(a.dataset);rid=a.run_id or ('v3-'+a.action+'-'+str(time.time_ns()))
    out=a.output or ROOT/'C.results'/('1.train' if a.action=='train' else '3.infer' if a.action=='infer' else '2.test')/rid
    if Path(rid).name!=rid or rid in ['.','..']:ap.error('Invalid --run-id')
    out=out.resolve()
    if not out.is_relative_to(ROOT):ap.error('--output must be inside SARdetect for portable candidate assets')
    if out.exists() and not os.environ.get('SAR_TRAIN_CONTEXT'):raise FileExistsError(f'Refusing to overwrite run: {out}')
    write(out/'request.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},models=keys,dataset=str(data),manifest_sha256=sha(ROOT/'B.models/v3_manifest.json'),pid=os.getpid()))
    if a.checkpoint:
        if len(keys)!=1:ap.error('--checkpoint requires exactly one model')
        cp=a.checkpoint.resolve(strict=True);key=keys[0]
        cfg=manifest()
        if key=='model-c-s':
            if cp.suffix!='.json':ap.error('Classifier needs candidate_manifest.json covering all branches')
            cfg=read(cp)
            if 'classifier' not in cfg or cfg['classifier'].get('kind')!='leaf-ensemble-knn':ap.error('Invalid classifier manifest')
        else:
            row=cfg['refiner'] if key=='model-d-s' else cfg['detectors'][key]
            row.update(checkpoint=str(cp),sha256=sha(cp))
        override=out/'effective_manifest.json';write(override,cfg);os.environ['SARDETECT_MANIFEST']=str(override)
        write(out/'checkpoint.json',dict(path=str(cp),sha256=sha(cp)))
    if a.dry_run:
        write(out/'completed.json',dict(status='dry-run',models=keys));print(out);return
    if a.action=='train':
        from cascade_training import train
        for key in keys:
            from training_control import already_completed
            if already_completed(out/key):continue
            if key=='model-c-s':
                train_classifier=load_model_module('model-c-s','train_classifier').train
                train_classifier(key,data,out/key,a)
            else:train(key,data,out/key,a)
    else:
        from competition_metrics import evaluate
        coco=read(data/f'annotations/{a.split}.json')
        count=a.samples or (12 if a.action=='infer' else None)
        if count:
            coco['images']=coco['images'][:count];ids={im['id'] for im in coco['images']};coco['annotations']=[p for p in coco['annotations'] if p['image_id'] in ids]
        pipeline=Pipeline(depth_prior=a.depth_prior,output_depth=a.output_depth);results={}
        for key in keys:
            dest=out/key;dest.mkdir(parents=True,exist_ok=True)
            if key in ['model-c-s','model-d-s']:
                geometry=load_model_module('model-c-s','hierarchy_model')
                crop,geometry_quad,class_metrics=geometry.crop,geometry.geometry_quad,geometry.class_metrics
                from competition_metrics import poly_iou
                from types import SimpleNamespace
                torch=pipeline.torch;images={im['id']:im for im in coco['images']};rows=[];ious=[]
                with torch.no_grad():
                    for start in range(0,len(coco['annotations']),128):
                        batch=coco['annotations'][start:start+128]
                        prepared=[crop(cv2.imread(str(data/'images'/images[p['image_id']]['file_name']),0),p['bbox']) for p in batch]
                        yy=pipeline.models['refiner'](torch.stack([p[0] for p in prepared]).to('npu:0')).cpu().numpy()
                        for p,g,(_,rect) in zip(batch,yy,prepared):
                            quad=geometry_quad(g,rect);rows.append(dict(p,score=1.,quad=quad));ious.append(poly_iou(quad,p['quad']))
                if key=='model-c-s':
                    predictions=pipeline.classify(rows,coco,data/'images');q=np.asarray([p['probabilities'] for p in predictions])
                    result=class_metrics(q,SimpleNamespace(rows=coco['annotations']),pipeline.tax,0,prior=a.depth_prior,output_depth=pipeline.definition['classifier'].get('output_depth'));result['depth_prior']=a.depth_prior;write(dest/'predictions.json',predictions)
                else:result=dict(mean_iou=float(np.mean(ious)),recall_iou30=float(np.mean(np.asarray(ious)>=.3)))
                result.update(protocol='GT HBB proposal component diagnostic; not full pipeline detection',targets=len(rows));write(dest/'metrics.json',result);results[key]=result;continue
            hbb=read(a.hbb_predictions) if a.hbb_predictions else detect(key,data,coco,dest,a.split)
            ids={im['id'] for im in coco['images']};hbb=[p for p in hbb if p['image_id'] in ids]
            threshold=manifest()['detectors'][key]['threshold'];rows=pipeline(hbb,coco,data/'images',threshold)
            write(dest/'predictions.json',rows)
            from obb_labels import export_predictions,load_class_map
            original=read(data/'VERIFIED.json').get('source',str(data)) if (data/'VERIFIED.json').exists() else str(data)
            export_predictions(coco,rows,dest/'labels',load_class_map(original))
            cfg=pipeline.definition['classifier']
            modes=[('mixed','mixed')]+([('3','full_depth')] if all(r.get('l3',-1)>=0 for r in coco['annotations']) else [])
            if os.environ.get('SARDETECT_UNLABELED')=='1':
                modes=[];results[key]=dict(images=len(coco['images']),detections=len(rows),labels_available=False)
            for depth,name in modes:
                metric=evaluate(coco,rows,pipeline.tax,threshold,cfg['margin'],depth,tuple(cfg['depth_prior']),cfg.get('output_depth'));write(dest/f'metrics_{name}.json',metric)
                if depth=='mixed':results[key]={k:metric[k] for k in ['det_score','class_score','joint_score','tp','fp','fn','images']}
            if a.action=='infer':
                from PIL import Image,ImageDraw
                for im in coco['images']:
                    image=Image.open(data/'images'/im['file_name']).convert('RGB');draw=ImageDraw.Draw(image)
                    for p in rows:
                        if p['image_id']!=im['id']:continue
                        quad=[tuple(q) for q in p['quad']];draw.line(quad+[quad[0]],fill='red',width=2);draw.text(quad[0],p['label'],fill='yellow')
                    image.save(dest/f"sample_{im['id']}.png")
        write(out/'summary.json',results);print(json.dumps(results,indent=2))
    write(out/'completed.json',dict(status='passed',finished=time.time(),models=keys));print(out)
if __name__=='__main__':main()
