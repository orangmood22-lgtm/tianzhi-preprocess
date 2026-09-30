"""Installed cascade inference: HBB detector -> learned OBB -> hierarchical ensemble."""
import collections, hashlib, json, os, subprocess, sys
from pathlib import Path
import cv2
import numpy as np

from project_io import ROOT,UTIL,PYTHON,YOLO_PYTHON,read,write,sha,manifest,checked
from model_loader import load_model_module

def dataset(value=None):
    p=Path(value) if value else ROOT/manifest()['dataset']
    if not p.is_absolute():p=ROOT/'A.dataset'/p
    if (p/'taxonomy.json').exists() and read(p/'taxonomy.json')!=read(ROOT/manifest()['taxonomy']):raise ValueError('Dataset taxonomy differs from installed 4/21/40 schema')
    if (p/'labels').is_dir() or not all((p/'annotations'/f'{split}.json').is_file() for split in ('train','val','test')):
        from dataset_adapter import prepare
        p=prepare(p,ROOT/manifest()['taxonomy'])
    if read(p/'taxonomy.json')!=read(ROOT/manifest()['taxonomy']):raise ValueError('Dataset taxonomy differs from installed 4/21/40 schema')
    return p
def run(cmd,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    write(log.with_suffix('.command.json'),dict(command=list(map(str,cmd)),pid=os.getpid(),device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES')))
    from training_control import tee
    tee(cmd,log)
def detect(key,data,coco,out,split='test'):
    row=manifest()['detectors'][key];raw=read(ROOT/row['config']);cp=checked(row)
    raw['model']['pretrained_detector']=str(cp)
    out=Path(out);raw['run']=dict(id='detector',output_root=str(out/'raw'))
    raw['data'].update(root=str(out/'annotations'),image_prefix=str(data/'images')+'/',**{f'{split}_annotations':f'{split}.json'})
    for name in ['train','val','test']:
        write(out/f'annotations/{name}.json',coco if name==split else read(data/f'annotations/{name}.json'))
    cfg=out/'resolved.json';write(cfg,raw)
    kind=raw['backend'];script={'faster':'faster_evaluate.py','openmmlab':'openmmlab_backend.py','rtdetr':'rtdetr_backend.py','yolo':'yolo_backend.py'}[kind]
    cmd=[YOLO_PYTHON if kind=='yolo' else PYTHON,UTIL/script,'--config',cfg,'--checkpoint',cp,'--run-id','detector','--split',split,'--dump-predictions']
    cmd+=['--output-root',out/'raw'] if kind=='faster' else ['--mode','test']
    if kind in ['rtdetr','yolo']:cmd+=['--device','npu:0']
    run(cmd,out/'detector.log')
    from result_artifacts import _coco_predictions
    preds=_coco_predictions(out/'raw/detector',coco);write(out/'hbb.json',preds)
    return preds

class Pipeline:
    def __init__(self,depth_prior=None,output_depth=None):
        import torch,torch_npu
        Heads=load_model_module('model-c-s','hierarchy_model').Heads
        self.torch=torch;torch.set_num_threads(4);torch.npu.set_device(0);cv2.setNumThreads(0)
        self.definition=manifest()
        from competition_metrics import validate_depth_prior
        self.definition['classifier']['depth_prior']=list(validate_depth_prior(depth_prior if depth_prior is not None else self.definition['classifier']['depth_prior']))
        if output_depth is not None:
            if output_depth not in (1,2,3):raise ValueError('output_depth must be 1, 2, or 3')
            self.definition['classifier']['output_depth']=output_depth
        self.tax=read(ROOT/self.definition['taxonomy']);self.models={}
        entries=[('refiner',self.definition['refiner'])]
        if self.definition['classifier'].get('kind')!='leaf-ensemble-knn':entries+=list(self.definition['classifier']['branches'].items())
        for key,row in entries:
            OBBRefiner=load_model_module('model-d-s','obb_model').OBBRefiner
            net=(OBBRefiner(self.tax) if key=='refiner' else Heads(self.tax,'classifier')).to('npu:0')
            net.load_state_dict(torch.load(checked(row),map_location='cpu',weights_only=False)['model']);net.eval();self.models[key]=net
    def refine(self,preds,coco,image_root):
        geometry=load_model_module('model-c-s','hierarchy_model')
        crop,geometry_quad=geometry.crop,geometry.geometry_quad
        from competition_metrics import poly_iou
        torch=self.torch;groups=collections.defaultdict(list);images={im['id']:im for im in coco['images']}
        for p in preds:
            if p['score']>=.02 and p['bbox'][2]>1 and p['bbox'][3]>1:groups[p['image_id']].append(p)
        result=[]
        with torch.no_grad():
            for ident,pp in groups.items():
                arr=cv2.imread(str(image_root/images[ident]['file_name']),0)
                if arr is None:raise FileNotFoundError(images[ident]['file_name'])
                current=[]
                for start in range(0,len(pp),64):
                    good=[];xx=[];rects=[]
                    for p in pp[start:start+64]:
                        try:x,rect=crop(arr,p['bbox'])
                        except (AssertionError,cv2.error):continue
                        good.append(p);xx.append(x);rects.append(rect)
                    if not good:continue
                    y=self.models['refiner'](torch.stack(xx).to('npu:0')).cpu().numpy()
                    for p,g,rect in zip(good,y,rects):current.append(dict(image_id=ident,score=p['score'],bbox=p['bbox'],quad=geometry_quad(g,rect)))
                kept=[]
                for p in sorted(current,key=lambda p:-p['score']):
                    if all(poly_iou(p['quad'],q['quad'])<.5 for q in kept):kept.append(p)
                result.extend(kept)
        return result
    def classify(self,rows,coco,image_root):
        pixels=load_model_module('model-c-s','crop_processing').pixels
        from competition_metrics import actions,decode,action_label
        if self.definition['classifier'].get('kind')=='leaf-ensemble-knn':
            ClassifierEnsemble=load_model_module('model-c-s','ensemble').ClassifierEnsemble
            if not hasattr(self,'phase30'):self.phase30=ClassifierEnsemble(ROOT,self.tax,self.definition['classifier']['assets'])
            q=self.phase30.predict(rows,coco,image_root);cfg=self.definition['classifier'];indices,_=decode(q,self.tax,cfg['margin'],cfg['depth_prior'],cfg.get('output_depth'));aa=actions(self.tax)
            return [dict(p,label=action_label(aa[i],self.tax),probabilities=pv.tolist(),label_depth=int(aa[i][0]),label_ids=list(aa[i][1]),leaf_top1=int(pv.argmax()),category_id=int(aa[i][1][aa[i][0]-1]),category_level=int(aa[i][0])) for p,pv,i in zip(rows,q,indices)]
        torch=self.torch;images={im['id']:im for im in coco['images']};arrays={};branches=[]
        if not rows:return []
        for ident in {p['image_id'] for p in rows}:
            arrays[ident]=cv2.imread(str(image_root/images[ident]['file_name']),0)
            if arrays[ident] is None:raise FileNotFoundError(images[ident]['file_name'])
        with torch.no_grad():
            for name,definition in self.definition['classifier']['branches'].items():
                outs=[];net=self.models[name]
                for start in range(0,len(rows),128):
                    batch=rows[start:start+128]
                    xx=np.stack([pixels(arrays[p['image_id']],p['bbox'],p['quad'],definition['mode']) for p in batch])
                    x=torch.from_numpy(xx[:,None]).to('npu:0');outs.append(net.log_probs(net(x))[2].exp().cpu().numpy())
                branches.append(np.concatenate(outs).astype(np.float64))
        # Research baseline probabilities came from JSON (float64); preserve its averaging dtype.
        q=np.mean(branches,axis=0);cfg=self.definition['classifier']
        q=np.maximum(q,1e-12)**(1/cfg['temperature']);q/=q.sum(1,keepdims=True)
        indices,_=decode(q,self.tax,cfg['margin'],cfg['depth_prior'],cfg.get('output_depth'));aa=actions(self.tax)
        result=[]
        for p,pv,index in zip(rows,q,indices):
            depth,node=aa[index]
            result.append(dict(p,label=action_label((depth,node),self.tax),probabilities=pv.tolist(),label_depth=int(depth),label_ids=list(node),leaf_top1=int(pv.argmax()),category_id=int(node[depth-1]),category_level=int(depth)))
        return result
    def __call__(self,preds,coco,image_root,threshold):
        rows=[p for p in self.refine(preds,coco,image_root) if p['score']>=threshold]
        return self.classify(rows,coco,image_root)
