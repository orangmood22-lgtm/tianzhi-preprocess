import math,random
import cv2,numpy as np
from project_io import ROOT,read,write,sha,manifest,checked
from model_loader import load_model_module
from training_control import EpochSession
def train_heads(key,data,out,args):
    import torch,torch_npu
    from torch import nn
    geometry=load_model_module('model-c-s','hierarchy_model')
    Heads,crop,target_geometry,geometry_quad,class_metrics=geometry.Heads,geometry.crop,geometry.target_geometry,geometry.geometry_quad,geometry.class_metrics
    pixels=load_model_module('model-c-s','crop_processing').pixels
    from competition_metrics import poly_iou
    torch.set_num_threads(4);cv2.setNumThreads(0);torch.npu.set_device(0)
    tax=read(data/'taxonomy.json');definition=manifest();definition['classifier']['depth_prior']=list(args.depth_prior);kind='refiner' if key=='model-d-s' else 'classifier'
    names=['refiner'] if kind=='refiner' else list(definition['classifier']['branches']) if args.branch=='all' else [args.branch]
    train_coco=read(data/'annotations/train.json');val_coco=read(data/'annotations/val.json')
    train_rows=train_coco['annotations'][:32] if args.smoke else train_coco['annotations'];val_rows=val_coco['annotations'][:32] if args.smoke else val_coco['annotations']
    from training_resources import ImageCache
    images={s:ImageCache({im['id']:data/'images'/im['file_name'] for im in coco['images']},64) for s,coco in [('train',train_coco),('val',val_coco)]}
    defaults=read(ROOT/'D.codes/1.model/modelds-OBBRefiner/config.json')['train']
    batch=args.batch_size or defaults['batch_size'];epochs=args.epochs or (1 if args.smoke else defaults['epochs'])
    lr=args.learning_rate or defaults['learning_rate']
    if min(batch,epochs,lr)<=0:raise ValueError('Training parameters must be positive')
    branches={}
    class RegularizedHeads(Heads):
        def forward(self,x):
            if self.kind=='refiner':return super().forward(x)
            f=nn.functional.dropout(self.features(x).mean((2,3)),.2,self.training)
            return self.h1(f),self.h2(f),self.h3(f)
    # Validation aligned views use learned geometry, never validation GT orientation.
    val_quads={}
    if kind=='classifier' and any(definition['classifier']['branches'][name]['mode']=='aligned' for name in names):
        ref=Heads(tax,'refiner').to('npu:0');ref.load_state_dict(torch.load(checked(definition['refiner']),map_location='cpu',weights_only=False)['model']);ref.eval()
        with torch.no_grad():
            for start in range(0,len(val_rows),batch):
                rows=val_rows[start:start+batch];prepared=[crop(images['val'][r['image_id']],r['bbox']) for r in rows]
                yy=ref(torch.stack([p[0] for p in prepared]).to('npu:0')).cpu().numpy()
                for r,g,(_,rect) in zip(rows,yy,prepared):val_quads[r['id']]=geometry_quad(g,rect)
        del ref
    for name in names:
        seed=3924 if name=='aligned3924' and args.seed==3923 else args.seed
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.npu.manual_seed_all(seed)
        row=definition['refiner'] if kind=='refiner' else definition['classifier']['branches'][name];cp=checked(row);mode=row.get('mode','hbb')
        dest=out/name;dest.mkdir(exist_ok=True);net=RegularizedHeads(tax,kind).to('npu:0');net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
        optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.02 if kind=='refiner' else .05)
        write(dest/'config.json',dict(depth_prior=list(args.depth_prior),kind=kind,mode=mode,seed=seed,epochs=epochs,batch_size=batch,learning_rate=lr,parent=str(cp),sha256=sha(cp),selection='val mean OBB IoU' if kind=='refiner' else 'val GT HBB proposal per-image hierarchy score; predicted OBB alignment',dataset_manifest=read(data/'VERIFIED.json')['manifest_sha256'],automatic_deployment=False))
        session=EpochSession(key,name,net,optimizer,epochs,dict(model_name=key,dataset=str(data),train_targets=len(train_rows),val_targets=len(val_rows),batch_size=batch,learning_rate=lr,seed=seed))
        history=session.history;best=session.best;stale=0;weights=session.path
        for epoch in range(session.start_epoch,epochs):
            net.train();order=np.random.permutation(len(train_rows));total=0.;seen=0
            for group in optimizer.param_groups:group['lr']=lr*(.2+.8*.5*(1+math.cos(math.pi*epoch/epochs)))
            for start in range(0,len(order),batch):
                rows=[train_rows[i] for i in order[start:start+batch]];xx=[];targets=[]
                for r in rows:
                    arr=images['train'][r['image_id']]
                    if kind=='refiner':
                        x,y,w,h=r['bbox'];bw=w*random.uniform(.85,1.2);bh=h*random.uniform(.85,1.2)
                        box=[x+w/2+random.uniform(-.12,.12)*w-bw/2,y+h/2+random.uniform(-.12,.12)*h-bh/2,bw,bh]
                        t,rect=crop(arr,box);xx.append(t.numpy()[0]);targets.append(target_geometry(r['quad'],rect))
                    else:
                        x=pixels(arr,r['bbox'],r['quad'],mode)
                        if random.random()<.5:x=x[:,::-1]
                        if random.random()<.5:x=x[::-1]
                        x=cv2.warpAffine(x,np.float32([[1,0,random.uniform(-6,6)],[0,1,random.uniform(-6,6)]]),(160,160))
                        x=np.clip(x,0,1)**random.uniform(.7,1.4)*random.uniform(.8,1.2)
                        if random.random()<.5:x*=1+np.random.normal(0,random.uniform(0,.12),x.shape).astype(np.float32)
                        xx.append(np.clip(x,0,1));targets.append([r['l1'],r['l2'],r['l3']])
                x=torch.from_numpy(np.stack(xx)[:,None]).float().to('npu:0');y=torch.as_tensor(np.array(targets),device='npu:0');optimizer.zero_grad(set_to_none=True);pred=net(x)
                if kind=='refiner':loss=nn.functional.smooth_l1_loss(pred[:,:4],y[:,:4])+.3*nn.functional.mse_loss(nn.functional.normalize(pred[:,4:],dim=1),y[:,4:])
                else:loss=sum(w*(.95*nn.functional.nll_loss(p,y[:,i])+.05*(-p.mean())) for i,(w,p) in enumerate(zip([1,.8,1],net.log_probs(pred))))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step();total+=float(loss.detach().cpu())*len(rows);seen+=len(rows)
                session.progress(epoch,start//batch+1,(len(order)+batch-1)//batch,float(loss.detach().cpu()))
            if not session.validate(epoch):
                session.finish(epoch,total/seen,{},None,dict(taxonomy=tax));write(dest/'history.json',session.history);continue
            net.eval();outputs=[]
            with torch.no_grad():
                for start in range(0,len(val_rows),batch):
                    rows=val_rows[start:start+batch]
                    if kind=='refiner':
                        prepared=[crop(images['val'][r['image_id']],r['bbox']) for r in rows];pred=net(torch.stack([p[0] for p in prepared]).to('npu:0')).cpu().numpy()
                        outputs.extend([poly_iou(geometry_quad(g,rect),r['quad']) for r,g,(_,rect) in zip(rows,pred,prepared)])
                    else:
                        xx=np.stack([pixels(images['val'][r['image_id']],r['bbox'],val_quads.get(r['id']),mode) for r in rows]);x=torch.from_numpy(xx[:,None]).to('npu:0');outputs.extend(net.log_probs(net(x))[2].exp().cpu().numpy())
            if kind=='refiner':metric=dict(mean_iou=float(np.mean(outputs)));score=metric['mean_iou']
            else:
                from types import SimpleNamespace
                metric=class_metrics(np.asarray(outputs),SimpleNamespace(rows=val_rows),tax,0,prior=args.depth_prior);score=metric['mixed_score']
            improved=score>best;best=max(best,score);stale=0 if improved else stale+1
            session.finish(epoch,total/seen,metric,score,dict(taxonomy=tax));write(dest/'history.json',session.history)
        net.load_state_dict(torch.load(weights/'best.pth',map_location='cpu',weights_only=False)['model']);del net
        branches[name]=dict(mode=mode,checkpoint=str(weights/'best.pth'),sha256=sha(weights/'best.pth'),best=best,reload_verified=True)
    write(out/'candidate.json',dict(status='candidate',model=key,branches=branches,gate=definition['classifier'],automatic_deployment=False,note='Warm-start fine-tuning; validate the combined ensemble and select gate only on val before publication.'))
