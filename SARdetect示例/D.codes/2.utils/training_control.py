"""Run naming, observable training, and epoch-boundary recovery for SARdetect."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def naming(dataset, models, config='00', stamp=None):
    dataset = re.sub(r'[^\w-]+', '-', Path(dataset).name)
    model = '-'.join(k.replace('model-', 'model').replace('-s', 's') for k in models)
    return f'{dataset}_{model}_cfg{config.removeprefix("config-")}_{stamp or dt.datetime.now().strftime("%Y%m%d%H%M%S")}'

def valid_name(name):
    return bool(re.fullmatch(r'[\w-]+_model[\w-]+_cfg\d{2}_\d{8}(?:\d{6})?', name))

def context():
    return read(os.environ['SAR_TRAIN_CONTEXT']) if os.environ.get('SAR_TRAIN_CONTEXT') else None

def checkpoint_dir(key=None, branch=None):
    c = context()
    if not c: raise RuntimeError('Use SARdetect/train.py for managed training')
    p = Path(c['checkpoint_dir']) / (key or os.environ['SAR_TRAIN_MODEL'])
    if branch: p /= branch
    p.mkdir(parents=True, exist_ok=True)
    return p

def event(kind, **values):
    c = context()
    if not c: return
    record = dict(time=dt.datetime.now().isoformat(), kind=kind, model=os.environ.get('SAR_TRAIN_MODEL'), **values)
    line = json.dumps(record, ensure_ascii=False, default=str)
    print('[SAR] ' + line, flush=True)
    with (Path(c['run_dir']) / 'events.jsonl').open('a', encoding='utf-8') as f: f.write(line + '\n')
    if kind=='configuration' and c.get('public_dir'):
        name=(record.get('model') or 'model')+('-'+record['branch'] if record.get('branch') else '')
        write(Path(c['public_dir'])/'config'/(name+'.json'),{k:v for k,v in record.items() if k not in ('time','kind')})

def tee(cmd, log, process_group=False, public=None):
    cmd = list(map(str, cmd)); log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    with log.open('a', encoding='utf-8') as f:
        f.write('\nCOMMAND ' + json.dumps(cmd) + '\n'); f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=log.parent, env=env, text=True, encoding='utf-8', errors='replace', bufsize=1, start_new_session=process_group)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True); f.write(line); f.flush()
                if public:
                    from experiment_output import append_line
                    append_line(public,line)
            code = proc.wait()
        except BaseException:
            import signal
            if process_group:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            else:proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if process_group:os.killpg(proc.pid,signal.SIGKILL)
                else:proc.kill()
                proc.wait()
            raise
    if code: raise subprocess.CalledProcessError(code, cmd)

def fingerprint(data):
    h = hashlib.sha256(); count = 0
    files = sorted(set(list(data.glob('*.txt')) + list(data.glob('taxonomy.json')) + list(data.glob('class_map.json')) + list((data/'annotations').glob('*.json')) + list((data/'labels').rglob('*.txt'))))
    for p in files:
        if p.name == 'data.txt': continue
        h.update(str(p.relative_to(data)).encode()); h.update(p.read_bytes()); count += 1
    for p in sorted((data/'images').rglob('*')):
        if p.is_file(): h.update(f'{p.relative_to(data)}:{p.stat().st_size}:{p.stat().st_mtime_ns}'.encode())
    if not count: raise ValueError('Dataset has no split/annotation files: ' + str(data))
    return h.hexdigest()

def dataset_scale(data):
    result={}
    for split in ['train','val','test']:
        if (data/f'annotations/{split}.json').is_file():
            coco=read(data/f'annotations/{split}.json'); rows=coco['annotations']
            result[split]=dict(images=len(coco['images']),targets=len(rows),label_depths={str(d):sum(r.get('mixed_label_depth',3)==d for r in rows) for d in [1,2,3]})
        else:
            from yolo_data import read_split
            images=read_split(data,split); count=0
            for image in images:
                lab=data/'labels'/image.relative_to((data/'images').resolve()).with_suffix('.txt')
                count+=sum(bool(line.strip()) and not line.lower().startswith(('#','imagesource:','gsd:')) for line in lab.read_text(encoding='utf-8-sig').splitlines())
            result[split]=dict(images=len(images),targets=count)
    return result

def resolve_training_script(script):
    """Resolve old saved entry paths without rewriting historical training records."""
    script = {'sar_v3/cli.py': 'cascade_cli.py'}.get(script, script)
    if script not in {'cascade_cli.py', 'sived_runtime.py', 'legacy_components_cli.py'}:
        raise ValueError('Unregistered saved training entry: ' + str(script))
    return ROOT/'D.codes/2.utils'/script


def supervise(script, argv):
    """Called before framework initialization; owns lock, request and attempt records."""
    from experiment_output import work_dir, public_dir, resolve_run, initialize, training_request, publish_training
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--resume', type=Path); p.add_argument('--val-interval', type=int, default=1)
    p.add_argument('--save-interval', type=int, default=1); p.add_argument('--log-interval', type=int, default=20)
    a, rest = p.parse_known_args(argv)
    if min(a.val_interval, a.save_interval, a.log_interval) < 1: raise ValueError('Intervals must be positive')
    if a.resume:
        if rest: raise ValueError('Resume uses the saved configuration: python train.py --resume <run directory>')
        run = resolve_run(a.resume)
        c = read(run/'training.json'); script = c['script']; rest = c['argv']
        if fingerprint(Path(c['dataset'])) != c['dataset_fingerprint']: raise ValueError('Dataset changed since the original training run')
        c['resume'] = True
        c.pop('restart_from_epoch_zero',None)
        if c.get('status') == 'completed':
            if c.get('report_error'):
                publish_training(c);c.pop('report_error');write(run/'training.json',c)
                print('Training was already completed; repaired result publication without retraining.')
                return
            raise ValueError('Training already completed; start a new run to change its schedule')
        if not any(Path(c['checkpoint_dir']).rglob('*.pth')) and not any(Path(c['checkpoint_dir']).rglob('*.pt')):
            # An interrupted queue may include a model still initializing its
            # first epoch. Its saved request can safely restart from epoch zero.
            c['resume'] = False
            c['restart_from_epoch_zero'] = True
            print('[SAR] No completed checkpoint for this model; restarting the saved request at epoch 0.',flush=True)
    else:
        q = argparse.ArgumentParser(add_help=False)
        q.add_argument('action'); q.add_argument('--models', nargs='+'); q.add_argument('--dataset', required=True)
        q.add_argument('--run-id'); q.add_argument('--output', type=Path); q.add_argument('--config', default='00'); q.add_argument('--time-code')
        q.add_argument('--epochs',type=int);q.add_argument('--batch-size',type=int);q.add_argument('--learning-rate',type=float)
        b, _ = q.parse_known_args(rest)
        import math
        for field in ['epochs','batch_size','learning_rate']:
            value=getattr(b,field)
            if value is not None and (not math.isfinite(value) or value<=0):raise ValueError(field+' must be positive and finite')
        name = b.run_id or naming(b.dataset, b.models, b.config, b.time_code)
        if b.output: name = b.output.name
        if not valid_name(name): raise ValueError('Run name must be <dataset>_modelN_cfg00_YYYYMMDD[HHMMSS]')
        if name.rsplit('_',1)[0]!=naming(b.dataset,b.models,b.config,'00000000').rsplit('_',1)[0]:raise ValueError('Run name must match the selected dataset, model and config')
        public = (ROOT/'C.results/1.train'/name).resolve()
        if b.output and b.output.resolve() != public: raise ValueError('Training output must be C.results/1.train/<standard run name>')
        run=work_dir(public)
        if run.exists() or public.exists(): raise FileExistsError('Existing run; use --resume ' + str(public))
        data = (ROOT/'A.dataset'/b.dataset).resolve(strict=True)
        # Remove routing-only options, then give the original backend its canonical directory.
        for flag in ['--output', '--run-id', '--time-code']:
            if flag in rest: i=rest.index(flag); del rest[i:i+2]
        rest += ['--output', str(run)]
        c = dict(version=2, run_dir=str(run), public_dir=str(public), checkpoint_dir=str(ROOT/'.runtime/checkpoints'/name), script=script, argv=rest, dataset=str(data), dataset_fingerprint=fingerprint(data), models=b.models, val_interval=a.val_interval, save_interval=a.save_interval, log_interval=a.log_interval, resume=False, created=dt.datetime.now().isoformat())
        c['full_dataset_scale']=dataset_scale(data)
        catalog=read(ROOT/'D.codes/1.model/catalog.json');c['architectures']={k:catalog[k]['architecture'] for k in b.models}
        run.mkdir(parents=True)
    public=Path(c.get('public_dir',public_dir(run)))
    if public==run:raise ValueError('Legacy run needs output migration before resume: '+str(public))
    initialize(public,training_request(c),'准备训练')
    # Store resource choices so resume does not silently revert to old workers.
    if not a.resume:
        c['resources'] = dict(workers=int(os.environ.get('SAR_TRAIN_WORKERS','0')))
        write(run/'initial_request.json', c)
    if 'resources' in c:
        os.environ['SAR_TRAIN_WORKERS'] = str(c['resources']['workers'])
    # Cross-queue lock: overlapping physical NPUs must not start two trainers.
    import fcntl
    locks = []
    device = '0'
    for flag in ['--device','--devices']:
        if flag in rest: device = rest[rest.index(flag)+1].removeprefix('npu:')
    for card in sorted(device.split(',')):
        if not card.isdigit(): raise ValueError('Numeric NPU IDs required')
        handle = (ROOT/'.runtime/locks'/('npu-'+card+'.lock'))
        handle.parent.mkdir(parents=True,exist_ok=True)
        handle = handle.open('a')
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('NPU '+card+' is already used by another managed training job')
        locks.append(handle)
    with (run/'training.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This run already has a live training process')
        attempts = list(run.glob('attempt-*.json')); attempt=len(attempts)+1
        for field in ['error','finished','exit_code','attempt_seconds']: c.pop(field,None)
        c.update(status='running', attempt=attempt, pid=os.getpid(), started=dt.datetime.now().isoformat()); write(run/'training.json', c)
        write(run/f'attempt-{attempt:03d}.json', c)
        import signal
        def interrupted(signum, frame): raise KeyboardInterrupt('Termination requested')
        signal.signal(signal.SIGTERM, interrupted)
        os.environ.update(SAR_TRAIN_CONTEXT=str(run/'training.json'), PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT/'D.codes/2.utils')+os.pathsep+os.environ.get('PYTHONPATH',''))
        print('SARdetect training\n' + json.dumps(c, ensure_ascii=False, indent=2), flush=True)
        started = time.monotonic(); code = 0
        try:
            tee([sys.executable, '-u', resolve_training_script(script), *rest], run/f'console-attempt-{attempt:03d}.log', process_group=True, public=public)
            c['status']='completed'
        except (KeyboardInterrupt, subprocess.CalledProcessError) as e:
            code = getattr(e, 'returncode', 130); c['status']='interrupted' if code in [-15,-9,-2,130,137,143] else 'failed'; c['error']=str(e)
        except Exception as e:
            code=1;c['status']='failed';c['error']=repr(e)
        finally:
            c.update(finished=dt.datetime.now().isoformat(), attempt_seconds=time.monotonic()-started, exit_code=code)
            write(run/'training.json', c); write(run/f'attempt-{attempt:03d}.json', c)
            try:publish_training(c)
            except Exception as error:
                c['report_error']=repr(error);write(run/'training.json',c)
                if not code:raise
                print('Result publication failed: '+repr(error),flush=True)
            print(f"[SAR] {c['status']}: records={public}/record.txt results={public}/result.txt", flush=True)
        if code: raise SystemExit(code)

def already_completed(path):
    c=context()
    return bool(c and c['resume'] and (Path(path)/'candidate.json').is_file())

def configure(raw, key):
    c = context()
    if not c: return
    os.environ['SAR_TRAIN_MODEL']=key
    tr=raw['train']; tr.update(val_interval=c['val_interval'], checkpoint_interval=c['save_interval'], log_interval=c['log_interval'])
    if 'resources' in c:
        tr.update(workers=c['resources']['workers'],val_batch_size=1,test_batch_size=1)
    raw['training_control']=dict(checkpoint_dir=str(checkpoint_dir(key)), resume=c['resume'])
    saved_config=Path(c['run_dir'])/key/'training_config.json'
    if c['resume'] and saved_config.exists():
        previous=read(saved_config)['raw']
        for field in ['train','model','data','runtime']:
            if raw[field]!=previous[field]:raise ValueError(f'Cannot resume: {field} configuration changed')
    sizes={}
    for split in ['train','val','test']:
        p=Path(raw['data']['root'])/raw['data'].get(split+'_annotations', split+'.json')
        coco=read(p); sizes[split]=dict(images=len(coco['images']), targets=len(coco['annotations']))
    event('configuration', model_name=key, architecture=raw['model'], dataset=c['dataset'], used_dataset=sizes, training=tr, seed=raw['runtime'].get('seed'), checkpoint_dir=str(checkpoint_dir(key)))
    write(Path(c['run_dir'])/key/'training_config.json', dict(raw=raw, sizes=sizes))

def rng_state():
    import numpy as np, torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), npu=torch.npu.get_rng_state_all())

def restore_rng(state):
    if not state: return
    import numpy as np, torch
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu()); torch.npu.set_rng_state_all([x.cpu() for x in state['npu']])

def save_torch(path, state):
    import torch
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp')
    torch.save(state,tmp); os.replace(tmp,path)

class EpochSession:
    """Component trainers save model, optimizer, RNG, best score and history together."""
    def __init__(self,key,branch,net,opt,epochs,details):
        import torch
        os.environ['SAR_TRAIN_MODEL']=key
        self.path=checkpoint_dir(key,branch); self.net=net; self.opt=opt; self.epochs=epochs; self.branch=branch
        self.start_epoch=0; self.best=-1.; self.history=[]; self.started=time.monotonic(); self.c=context(); self.pending=None; self.details=details; self.previous_seconds=0.
        if self.c['resume'] and (self.path/'last.pth').exists():
            state=torch.load(self.path/'last.pth',map_location='cpu',weights_only=False)
            if state.get('configuration',details)!=details:raise ValueError('Component training configuration changed')
            net.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); self.start_epoch=state['epoch']; self.best=state['best']; self.history=state['history']; self.pending=state['rng']
            self.previous_seconds=state.get('elapsed_seconds',0.)
            restore_rng(self.pending)
            event('resumed',branch=branch,completed_epoch=self.start_epoch,optimizer_restored=True,rng_restored=True)
        event('configuration',branch=branch,epochs=epochs,**details)
    @property
    def total_seconds(self): return self.previous_seconds+time.monotonic()-self.started
    def validate(self, epoch): return (epoch+1)%self.c['val_interval']==0 or epoch+1==self.epochs
    def progress(self,epoch,step,steps,loss):
        if step%self.c['log_interval'] and step!=steps: return
        done=(epoch-self.start_epoch)*steps+step; remain=(self.epochs-epoch-1)*steps+steps-step
        event('progress',branch=self.branch,epoch=epoch+1,epochs=self.epochs,step=step,steps=steps,loss=float(loss),eta_seconds=round((time.monotonic()-self.started)/max(done,1)*remain,1))
    def finish(self,epoch,loss,metric,score,extra=None):
        improved=score is not None and score>self.best
        if improved:self.best=score
        self.history.append(dict(epoch=epoch+1,loss=loss,**metric))
        state=dict(model={k:v.detach().cpu() for k,v in self.net.state_dict().items()},optimizer=self.opt.state_dict(),epoch=epoch+1,best=self.best,history=self.history,rng=rng_state(),configuration=self.details,elapsed_seconds=self.total_seconds,**(extra or {}))
        save_torch(self.path/'last.pth',state)
        if improved:save_torch(self.path/'best.pth',state)
        if (epoch+1)%self.c['save_interval']==0 or epoch+1==self.epochs:save_torch(self.path/f'epoch_{epoch+1:04d}.pth',state)
        if score is not None:event('validation',branch=self.branch,epoch=epoch+1,metrics=metric,best=self.best)
        event('checkpoint',branch=self.branch,epoch=epoch+1,path=str(self.path/'last.pth'),resumable=True)
