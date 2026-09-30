"""Managed SIVED OBB regression training and polygon evaluation for five cascades."""
import collections
import math
import random
from pathlib import Path
import cv2
import numpy as np
from project_io import ROOT, read, write, sha


def geometry_module():
    from model_loader import load_model_module
    return load_model_module('model-c-s','hierarchy_model')


def train(key, annotations, image_root, out, args, init):
    import torch, torch_npu
    from torch import nn
    from training_control import EpochSession
    from competition_metrics import poly_iou
    torch.set_num_threads(4); cv2.setNumThreads(0); torch.npu.set_device(0)
    seed=args.seed+int(key.split('-')[1]); random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.npu.manual_seed_all(seed)
    g=geometry_module(); net=g.Heads({},'refiner').to('npu:0')
    net.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['model'])
    rows={s:read(annotations/f'{s}.json') for s in ['train','val']}
    # Image IDs restart in each split: never key the image cache by ID alone.
    from training_resources import ImageCache
    images={s:ImageCache({im['id']:image_root/im['file_name'] for im in rows[s]['images']},64) for s in rows}
    assert all('quad' in r for c in rows.values() for r in c['annotations']), 'OBB training requires four-point labels'
    epochs=args.obb_epochs; batch=args.obb_batch_size; lr=args.obb_learning_rate
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.02)
    details=dict(model_name=key,branch_architecture='EfficientNet-B0 six-value OBB regression',train_targets=len(rows['train']['annotations']),val_targets=len(rows['val']['annotations']),seed=seed,planned_epochs=epochs,batch_size=batch,learning_rate=lr,parent=str(init),parent_sha256=sha(init),selection='val mean polygon IoU on GT HBB proposals; detection threshold selected separately on val')
    session=EpochSession(key,'obb',net,opt,epochs,details)
    write(out/'obb_config.json',details)
    for epoch in range(session.start_epoch,epochs):
        net.train(); order=np.random.permutation(len(rows['train']['annotations'])); total=0.; seen=0
        for group in opt.param_groups: group['lr']=lr*(.1+.9*.5*(1+math.cos(math.pi*epoch/epochs)))
        for start in range(0,len(order),batch):
            xx=[]; yy=[]
            for i in order[start:start+batch]:
                r=rows['train']['annotations'][i]; arr=images['train'][r['image_id']]
                x,y,w,h=r['bbox']; bw=w*random.uniform(.8,1.25); bh=h*random.uniform(.8,1.25)
                box=[x+w/2+random.uniform(-.15,.15)*w-bw/2,y+h/2+random.uniform(-.15,.15)*h-bh/2,bw,bh]
                t,rect=g.crop(arr,box); xx.append(t); yy.append(g.target_geometry(r['quad'],rect))
            x=torch.stack(xx).to('npu:0'); y=torch.from_numpy(np.asarray(yy)).to('npu:0')
            opt.zero_grad(set_to_none=True); pred=net(x)
            loss=nn.functional.smooth_l1_loss(pred[:,:4],y[:,:4])+.3*nn.functional.mse_loss(nn.functional.normalize(pred[:,4:],dim=1),y[:,4:])
            if not torch.isfinite(loss): raise ValueError('Nonfinite OBB loss')
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),2); opt.step()
            value=float(loss.detach().cpu()); total+=value*len(xx); seen+=len(xx)
            session.progress(epoch,start//batch+1,(len(order)+batch-1)//batch,value)
        metrics={}; score=None
        if session.validate(epoch):
            net.eval(); ious=[]
            with torch.no_grad():
                rr=rows['val']['annotations']
                for start in range(0,len(rr),batch):
                    part=rr[start:start+batch]; crops=[g.crop(images['val'][r['image_id']],r['bbox']) for r in part]
                    preds=net(torch.stack([c[0] for c in crops]).to('npu:0')).cpu().numpy()
                    ious.extend(poly_iou(g.geometry_quad(v,rect),r['quad']) for r,v,(_,rect) in zip(part,preds,crops))
            score=float(np.mean(ious)); metrics=dict(val_mean_polygon_iou=score,val_iou30_recall=float(np.mean(np.asarray(ious)>=.3)))
        session.finish(epoch,total/seen,metrics,score,dict(box_type='OBB',parameterization='dx dy logw logh cos2theta sin2theta'))
        write(out/'obb_history.json',session.history)
    cp=session.path/'best.pth'; check=torch.load(cp,map_location='cpu',weights_only=False); net.load_state_dict(check['model'])
    result=dict(checkpoint=str(cp),sha256=sha(cp),architecture='EfficientNet-B0 six-value OBB regression',epochs=epochs,best_epoch=check['epoch'],best_val_mean_iou=session.best,training_seconds=session.total_seconds,reload_verified=True)
    write(out/'obb_candidate.json',result)
    return result


def refine(coco, predictions, image_root, checkpoint, floor=.05, batch=64):
    import torch, torch_npu
    torch.set_num_threads(4); cv2.setNumThreads(0); torch.npu.set_device(0)
    g=geometry_module(); net=g.Heads({},'refiner').to('npu:0')
    net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']); net.eval()
    by=collections.defaultdict(list)
    for p in predictions:
        if p['score']>=floor: by[p['image_id']].append(p)
    output=[]
    with torch.no_grad():
        for im in coco['images']:
            arr=cv2.imread(str(image_root/im['file_name']),0)
            if arr is None: raise ValueError('Unreadable inference image')
            pp=by[im['id']]
            for start in range(0,len(pp),batch):
                part=pp[start:start+batch]; crops=[g.crop(arr,p['bbox']) for p in part]
                preds=net(torch.stack([c[0] for c in crops]).to('npu:0')).cpu().numpy()
                for p,v,(_,rect) in zip(part,preds,crops):
                    q=g.geometry_quad(v,rect); xs,ys=np.asarray(q).T
                    output.append(dict(**p,quad=q,obb_envelope=[float(xs.min()),float(ys.min()),float(np.ptp(xs)),float(np.ptp(ys))]))
    return output


def metric(coco, predictions, threshold):
    from competition_metrics import match
    gt=collections.defaultdict(list); pp=collections.defaultdict(list)
    for r in coco['annotations']:
        if 'quad' not in r: raise ValueError('OBB evaluation requires source polygons')
        gt[r['image_id']].append(r)
    for r in predictions:
        if r['score']>=threshold: pp[r['image_id']].append(r)
    rows=[]; sizes={k:dict(gt=0,tp=0) for k in ['small','medium','large']}
    for im in coco['images']:
        gg=gt[im['id']]; ps=pp[im['id']]; matches=match(ps,gg,.3); found={k for _,k,_ in matches}
        tp=len(matches); fp=len(ps)-tp; fn=len(gg)-tp
        for k,r in enumerate(gg):
            area=cv2.contourArea(np.asarray(r['quad'],np.float32)); name='small' if area<1024 else 'medium' if area<9216 else 'large'
            sizes[name]['gt']+=1; sizes[name]['tp']+=int(k in found)
        rows.append(dict(image_id=im['id'],file_name=im['file_name'],gt=len(gg),tp=tp,fp=fp,fn=fn,det_score=5*tp/(5*tp+fn+4*fp) if tp+fp+fn else 1.))
    values=np.asarray([r['det_score'] for r in rows]); rng=np.random.default_rng(3929)
    boot=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    totals={k:sum(r[k] for r in rows) for k in ['tp','fp','fn']}
    return dict(det_score=float(values.mean()),**totals,images=len(rows),threshold=float(threshold),recall=totals['tp']/max(totals['tp']+totals['fn'],1),precision=totals['tp']/max(totals['tp']+totals['fp'],1),per_image=rows,by_size=sizes,conditional_image_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),protocol='SIVED-OBB-v1: original four-point polygons; class-agnostic polygon IoU>=0.3; confidence-greedy one-to-one matching; mean per-image 5TP/(5TP+FN+4FP); empty correct=1; historically exposed test; no classification score')


def choose_threshold(coco, pp):
    trials=[metric(coco,pp,t) for t in np.arange(.05,1.,.05)]
    best=max(trials,key=lambda x:(x['det_score'],x['threshold']))
    return best, [{k:r[k] for k in ['threshold','det_score','tp','fp','fn']} for r in trials]
