"""Small lifecycle adapters; registered detection architectures remain unchanged."""
import time
from pathlib import Path
from training_control import context, event, rng_state, restore_rng, save_torch, write

def mmengine_config(cfg, raw):
    if 'training_control' not in raw: return
    # MMEngine's general file backend serializes to BytesIO and duplicates an
    # entire checkpoint in host RAM. Local competition runs can stream to disk.
    import importlib
    runner_module=importlib.import_module('mmengine.runner.runner')
    original_save=runner_module.save_checkpoint
    def local_save(checkpoint, filename, file_client_args=None, backend_args=None):
        if '://' in str(filename) or file_client_args or backend_args:
            return original_save(checkpoint,filename,file_client_args=file_client_args,backend_args=backend_args)
        save_torch(filename,checkpoint)
    runner_module.save_checkpoint=local_save
    ctl=raw['training_control']; folder=Path(ctl['checkpoint_dir']); folder.mkdir(parents=True,exist_ok=True)
    cfg['default_hooks']['checkpoint'].update(out_dir=str(folder), interval=raw['train']['checkpoint_interval'], save_last=True, max_keep_ckpts=2)
    cfg['train_cfg']['dynamic_intervals']=[(raw['train']['epochs'],1)]
    # CheckpointHook may append the work-dir basename to out_dir. Its pointer is authoritative.
    pointer=Path(cfg['work_dir'])/'last_checkpoint'
    if ctl['resume']:
        if pointer.exists(): cp=Path(pointer.read_text().strip())
        else:
            choices=list(folder.rglob('epoch_*.pth')); cp=max(choices,key=lambda p:p.stat().st_mtime) if choices else None
        if cp is None or not cp.is_file(): raise FileNotFoundError('No completed epoch checkpoint available to resume')
        cfg.update(resume=True,load_from=str(cp))
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
    @HOOKS.register_module(force=True)
    class SARHostMemoryHook(Hook):
        priority='HIGHEST'
        def after_train_epoch(self,runner):
            from training_resources import release_host_memory
            release_host_memory()
        def after_val_epoch(self,runner,metrics=None):
            from training_resources import release_host_memory
            release_host_memory()
    cfg.setdefault('custom_hooks',[]).append(dict(type='SARHostMemoryHook'))
    @HOOKS.register_module(force=True)
    class SARProgressHook(Hook):
        priority='VERY_LOW'
        def before_train(self,runner):
            self.started=time.monotonic(); self.start=runner.epoch
            event('resumed' if cfg.get('resume') else 'started',completed_epoch=runner.epoch,optimizer_restored=bool(cfg.get('resume')),scheduler_restored=bool(cfg.get('resume')))
        def before_save_checkpoint(self,runner,checkpoint): checkpoint['sar_rng']=rng_state()
        def after_load_checkpoint(self,runner,checkpoint):
            if cfg.get('resume'):restore_rng(checkpoint.get('sar_rng'))
        def after_train_epoch(self,runner):
            epoch=runner.epoch+1; eta=(time.monotonic()-self.started)/max(1,epoch-self.start)*(runner.max_epochs-epoch)
            event('progress',epoch=epoch,epochs=runner.max_epochs,eta_seconds=round(eta,1))
            if pointer.exists():event('checkpoint',epoch=epoch,path=pointer.read_text().strip(),resumable=True)
        def after_val_epoch(self,runner,metrics=None):event('validation',epoch=runner.epoch,metrics=metrics)
    cfg.setdefault('custom_hooks',[]).append(dict(type='SARProgressHook'))

def rtdetr_fit(solver,raw):
    import torch
    from src.solver.det_engine import train_one_epoch,evaluate
    from src.misc import dist_utils
    ctl=raw['training_control']; folder=Path(ctl['checkpoint_dir']);folder.mkdir(parents=True,exist_ok=True)
    solver.train(); args=solver.cfg; best=-1.; started=time.monotonic(); start=solver.last_epoch+1
    if ctl['resume']:
        old=torch.load(folder/'last.pth',map_location='cpu',weights_only=False);restore_rng(old.get('sar_rng'));best=old.get('sar_best',-1.)
    event('resumed' if ctl['resume'] else 'started',completed_epoch=start,optimizer_restored=ctl['resume'],scheduler_restored=ctl['resume'])
    for epoch in range(start,args.epoches):
        solver.train_dataloader.set_epoch(epoch)
        if dist_utils.is_dist_available_and_initialized():solver.train_dataloader.sampler.set_epoch(epoch)
        stats=train_one_epoch(solver.model,solver.criterion,solver.train_dataloader,solver.optimizer,solver.device,epoch,max_norm=args.clip_max_norm,print_freq=args.print_freq,ema=solver.ema,scaler=solver.scaler,lr_warmup_scheduler=solver.lr_warmup_scheduler,writer=solver.writer)
        if solver.lr_warmup_scheduler is None or solver.lr_warmup_scheduler.finished():solver.lr_scheduler.step()
        solver.last_epoch=epoch; improved=False; val={}
        if (epoch+1)%raw['train']['val_interval']==0 or epoch+1==args.epoches:
            val,_=evaluate(solver.ema.module if solver.ema else solver.model,solver.criterion,solver.postprocessor,solver.val_dataloader,solver.evaluator,solver.device)
            score=float(val['coco_eval_bbox'][0]);improved=score>best;best=max(best,score)
            event('validation',epoch=epoch+1,metrics=val,best=best)
        state=solver.state_dict();state.update(sar_rng=rng_state(),sar_best=best)
        save_torch(folder/'last.pth',state)
        if improved:save_torch(folder/'best.pth',state)
        if (epoch+1)%raw['train']['checkpoint_interval']==0 or epoch+1==args.epoches:save_torch(folder/f'epoch_{epoch+1:04d}.pth',state)
        write(Path(solver.output_dir)/f'epoch_{epoch+1:04d}.json',dict(epoch=epoch+1,train=stats,val=val))
        event('checkpoint',epoch=epoch+1,path=str(folder/'last.pth'),resumable=True)
        event('progress',epoch=epoch+1,epochs=args.epoches,eta_seconds=round((time.monotonic()-started)/(epoch+1-start)*(args.epoches-epoch-1),1))

def yolo_train(raw,kwargs,YOLO):
    """Keep full precision live model/optimizer, EMA, scaler, scheduler and RNG."""
    import torch
    from copy import deepcopy
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model
    import ultralytics.engine.trainer as trainer_module
    # The generic AMP probe downloads a separate YOLO model even in offline
    # mode. Use this registered training recipe's explicit AMP setting instead;
    # finite weights and effective optimizer updates are checked below.
    original_amp_check=trainer_module.check_amp
    trainer_module.check_amp=lambda model: bool(kwargs.get('amp',False))
    event('amp_configuration',enabled=bool(kwargs.get('amp',False)),source='registered training configuration',external_probe_download=False)
    ctl=raw['training_control'];folder=Path(ctl['checkpoint_dir']);folder.mkdir(parents=True,exist_ok=True)
    class SARTrainer(DetectionTrainer):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw)
            self.wdir=folder;self.last=folder/'last.pt';self.best=folder/'best.pt';self.sar_started=time.monotonic();self.sar_optimizer_steps=0
        def _setup_train(self):
            super()._setup_train()
            # A one-epoch small dataset can end before GradScaler backs off its
            # default 65536 scale. Keep the no-update guard and start short NPU
            # runs at a conservative scale; resumed scaler state stays intact.
            if not self.resume and self.epochs*len(self.train_loader)<100 and self.scaler.state_dict():
                state=self.scaler.state_dict()
                state['scale']=min(state['scale'],float(raw['train'].get('amp_initial_scale',128)))
                self.scaler.load_state_dict(state)
                event('configuration',branch='short-run-amp',initial_scale=state['scale'],
                      total_batches=self.epochs*len(self.train_loader),reason='short NPU run; optimizer update remains required')
            def counted_step(optimizer,args,kwargs):self.sar_optimizer_steps+=1
            self.optimizer.register_step_post_hook(counted_step)
        def setup_model(self):
            ckpt=super().setup_model()
            if self.resume and ckpt and ckpt.get('model') is not None:
                self.model.load_state_dict(ckpt['model'].float().state_dict())
            return ckpt
        def resume_training(self,ckpt):
            super().resume_training(ckpt)
            if self.resume:
                self.sar_optimizer_steps=ckpt.get('sar_optimizer_steps',0)
                self.scheduler.load_state_dict(ckpt['sar_scheduler']);restore_rng(ckpt['sar_rng'])
                event('resumed',completed_epoch=self.start_epoch,optimizer_restored=True,scheduler_restored=True,rng_restored=True)
            elif raw['train'].get('amp_initial_scale') and self.scaler.state_dict():
                state=self.scaler.state_dict();state['scale']=float(raw['train']['amp_initial_scale']);self.scaler.load_state_dict(state)
        def validate(self):
            if (self.epoch+1)%raw['train']['val_interval'] and self.epoch+1<self.epochs:
                return self.metrics, self.fitness if self.fitness is not None else 0.
            result=super().validate();event('validation',epoch=self.epoch+1,metrics=result[0]);return result
        def save_model(self):
            if not self.optimizer.state and self.epoch+1==self.epochs:raise RuntimeError('No effective optimizer update: inspect AMP/gradient accumulation before accepting this run')
            live=deepcopy(unwrap_model(self.model)).to('cpu',memory_format=torch.contiguous_format)
            ema=deepcopy(self.ema.ema).to('cpu',memory_format=torch.contiguous_format)
            for m in [live,ema]:
                if hasattr(m,'criterion'):m.criterion=None
            if not all(torch.isfinite(v).all() for v in live.state_dict().values() if v.is_floating_point()):raise ValueError('Nonfinite YOLO model state')
            state=dict(epoch=self.epoch,best_fitness=self.best_fitness,model=live,ema=ema,updates=self.ema.updates,optimizer=deepcopy(self.optimizer.state_dict()),scaler=self.scaler.state_dict(),train_args=vars(self.args),train_metrics={**self.metrics,'fitness':self.fitness},train_results=self.read_results_csv(),sar_scheduler=self.scheduler.state_dict(),sar_rng=rng_state(),sar_optimizer_steps=self.sar_optimizer_steps)
            save_torch(self.last,state)
            validated=(self.epoch+1)%raw['train']['val_interval']==0 or self.epoch+1==self.epochs
            if validated and (self.best_fitness==self.fitness or not self.best.exists()):save_torch(self.best,state)
            if (self.epoch+1)%raw['train']['checkpoint_interval']==0 or self.epoch+1==self.epochs:save_torch(folder/f'epoch_{self.epoch+1:04d}.pt',state)
            event('checkpoint',epoch=self.epoch+1,path=str(self.last),resumable=True,optimizer_state_entries=len(self.optimizer.state),optimizer_steps=self.sar_optimizer_steps)
            event('progress',epoch=self.epoch+1,epochs=self.epochs,eta_seconds=round((time.monotonic()-self.sar_started)/max(1,self.epoch+1-self.start_epoch)*(self.epochs-self.epoch-1),1))
            return True
        def final_eval(self):
            # Per-epoch validation already ran at the final epoch; do not strip resume state.
            event('completed',best_checkpoint=str(self.best),last_checkpoint=str(self.last))
    kwargs.update(exist_ok=True,patience=0,save_period=-1)
    cp=folder/'last.pt' if ctl['resume'] else Path(raw['model']['pretrained_detector'])
    if ctl['resume']:kwargs['resume']=str(cp)
    try:YOLO(str(cp)).train(trainer=SARTrainer,**kwargs)
    finally:trainer_module.check_amp=original_amp_check
