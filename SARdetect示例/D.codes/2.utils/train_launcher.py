"""Simple, resumable single/two-NPU queue for a host with 16 GiB RAM."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from project_io import ROOT
from training_control import dataset_scale, naming, read, write
from training_resources import interpreter, memory_available, process_tree_rss
from experiment_output import resolve_run

GIB = 1024**3


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return value


def memory_gb(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('Must be nonnegative; 0 uses host/container capacity')
    return value


def memory_reservation(model):
    # Conservative admission estimates; actual RSS is still monitored. Large
    # detectors should not overlap on a 16 GiB host, even with two free NPUs.
    return 12*GIB


def selection(models, epochs, catalog):
    names = [f'model-{i}' for i in range(1, 6)] if models == 'all' else models.split(',')
    aliases = {'cls': 'model-c-s', 'obb': 'model-d-s'}
    names = [aliases.get(n, n if n.startswith('model-') else 'model-'+n) for n in names]
    if not names or len(set(names)) != len(names) or any(n not in catalog for n in names):
        raise ValueError('Use 1..5, all, cls, obb, or registered model IDs; no duplicates')
    counts = [positive(n) for n in epochs.split(',')]
    if len(counts) == 1:
        counts *= len(names)
    if len(counts) != len(names):
        raise ValueError('Supply one epoch count or one count per selected model')
    return list(zip(names, counts))


def parser():
    p = argparse.ArgumentParser(description='SARdetect: python train.py DATASET MODELS EPOCHS; models=1,2,3,4,5 / all / cls / obb',
        epilog='Examples: python train.py SIVED_OBB 1 30 | python train.py SIVED_OBB all 30,40,30,40,50 | python train.py --resume last')
    p.add_argument('dataset', nargs='?')
    p.add_argument('models', nargs='?', default='all')
    p.add_argument('epochs', nargs='?', default='30')
    p.add_argument('--cards', default='0,1', help='Physical NPU IDs; default 0,1, serial jobs alternate cards')
    p.add_argument('--jobs', type=int, choices=range(1,9), default=1, help='Concurrent independent models, one per card; default serial')
    p.add_argument('--batch', type=positive, help='Per model batch; default detector=1, classifier/refiner=8')
    p.add_argument('--workers', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--obb-epochs', type=positive, default=12)
    p.add_argument('--every', type=positive, default=1, help='Validate and save every N epochs; always on final epoch')
    p.add_argument('--seed', type=int, default=3929)
    p.add_argument('--depth-prior', type=float, nargs=3)
    p.add_argument('--classifier-mode', choices=['mixed','level2'], default='mixed', help='Classifier training mode; level2 uses capped train labels and raw val labels')
    p.add_argument('--branch', choices=['all', 'baseline', 'aligned', 'aligned3924', 'convnext', 'dino'], default='all')
    p.add_argument('--memory-gb', type=memory_gb, default=12, help='Process-tree RSS budget; default 12 GiB; 0 uses host/container capacity minus 2 GiB reserve')
    p.add_argument('--smoke', action='store_true', help='Small training/validation subset; requested epochs still apply')
    p.add_argument('--dry-run', action='store_true', help='Print resolved jobs and dataset counts without starting training')
    p.add_argument('--check', action='store_true', help='Also check Python/torch-npu/NPU access; do not train')
    p.add_argument('--resume', type=str, help='Saved queue JSON, training directory, or last')
    return p


def make_plan(a):
    if not a.dataset:
        raise ValueError('Dataset required: python train.py DATASET MODELS EPOCHS')
    if Path(a.dataset).name != a.dataset or a.dataset in ['.', '..']:
        raise ValueError('Use a dataset folder name from A.dataset')
    # Registered datasets may be read-only links to the retained research data.
    data = (ROOT/'A.dataset'/a.dataset).resolve(strict=True)
    catalog = read(ROOT/'D.codes/1.model/catalog.json')
    selected = selection(a.models, a.epochs, catalog)
    cards = a.cards.split(',')
    if not 1 <= len(cards) <= 8 or len(set(cards)) != len(cards) or any(not c.isdigit() for c in cards):
        raise ValueError('--cards requires one to eight distinct physical NPU numbers')
    if a.jobs > len(cards):
        raise ValueError('--jobs must not exceed the number of cards')
    if a.depth_prior is not None:
        from competition_metrics import validate_depth_prior
        validate_depth_prior(a.depth_prior)
        if any(n != 'model-c-s' for n, _ in selected):
            raise ValueError('--depth-prior is for cls training')
    if a.branch != 'all' and any(n != 'model-c-s' for n, _ in selected):
        raise ValueError('--branch is for cls training')
    if a.classifier_mode!='mixed' and any(n != 'model-c-s' for n, _ in selected):
        raise ValueError('--classifier-mode is for cls training')
    stamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    queue_name = naming(a.dataset, [n for n, _ in selected], stamp=stamp)
    records = []
    for i, (model, epochs) in enumerate(selected):
        card = cards[i % len(cards)]
        run = ROOT/'C.results/1.train'/naming(a.dataset, [model], stamp=stamp)
        batch = a.batch or (8 if catalog[model]['role'] != 'detector' else 1)
        cmd = [interpreter(), '-u', str(ROOT/'D.codes/2.utils/competition_cli.py'), 'train',
               '--model', model, '--dataset', a.dataset, '--epochs', str(epochs),
               '--batch-size', str(batch), '--device', 'npu:'+card, '--seed', str(a.seed),
               '--val-interval', str(a.every), '--save-interval', str(a.every), '--run-id', run.name]
        if model in [f'model-{i}' for i in range(1, 6)]:
            cmd += ['--obb-epochs', str(a.obb_epochs), '--obb-batch-size', '8']
        if model == 'model-c-s':
            cmd += ['--branch', a.branch]
            if a.classifier_mode!='mixed':cmd += ['--classifier-mode',a.classifier_mode]
            if a.depth_prior is not None:
                cmd += ['--depth-prior', *map(str, a.depth_prior)]
        if a.smoke:
            cmd += ['--smoke']
        records.append(dict(model=model, architecture=catalog[model]['architecture'], epochs=epochs,
                            memory_reservation_bytes=memory_reservation(model),
                            batch_size=batch, card=card, run_dir=str(run), command=cmd, status='pending'))
    return dict(version=1, kind='training_queue', created=dt.datetime.now().isoformat(),
                path=str(ROOT/'.runtime/queues'/(queue_name+'.training_queue.json')),
                dataset=str(data), dataset_scale=dataset_scale(data), jobs=a.jobs, cards=cards,
                workers=a.workers, memory_budget_bytes=a.memory_gb*GIB, reserve_bytes=2*GIB,
                status='pending', records=records,
                note='jobs=2 trains independent models, one per NPU when memory reservations fit; large detectors serialize at the default 12 GiB budget. Not distributed training of one model.')


def preflight(plan):
    total, available = memory_available()
    if available < 4*GIB:
        raise RuntimeError('Less than 4 GiB available RAM; close other workloads first')
    if plan['jobs'] > 1 and available < 10*GIB:
        raise RuntimeError('Concurrent models require at least 10 GiB available RAM; use --jobs 1')
    limit = total-plan['reserve_bytes']
    plan['memory_budget_bytes'] = min(plan['memory_budget_bytes'] or limit, limit)
    plan['host_memory'] = dict(total_bytes=total, available_bytes=available)
    pythons = {interpreter()}
    if any(r['model'] in ['model-5', 'model-5-s'] for r in plan['records']):
        pythons.add(interpreter(True))
    probes = []
    for py in sorted(pythons):
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=','.join(plan['cards']))
        modules=['numpy','cv2','PIL','torchvision','yaml','matplotlib']
        if py==interpreter() and any(r['model'] in ['model-1','model-1-s','model-2','model-2-s','model-4','model-4-s'] for r in plan['records']):
            modules+=['mmengine','mmcv.ops']
        if py==interpreter() and any(r['model'] in ['model-1','model-1-s'] for r in plan['records']):
            modules+=['mmdet']
        if any(r['model'] not in ['model-c','model-c-s','model-d','model-d-s'] for r in plan['records']):
            modules+=['pycocotools']
        code = "import json,platform,importlib,torch,torch_npu; [importlib.import_module(m) for m in "+repr(modules)+"]; print(json.dumps(dict(python=platform.python_version(),arch=platform.machine(),torch=torch.__version__,torch_npu=torch_npu.__version__,npus=torch.npu.device_count(),checked_modules="+repr(modules)+"))); assert torch.npu.device_count()==" + str(len(plan['cards']))
        result = subprocess.run([py, '-c', code], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        print(result.stdout, flush=True)
        if result.returncode:
            raise RuntimeError('NPU environment check failed: '+py)
        probes.append(dict(python=py, output=result.stdout))
    plan['environment_checks'] = probes
    for row in plan['records']:
        row.setdefault('memory_reservation_bytes',memory_reservation(row['model']))


def stream(proc, model):
    with proc.stdout:
        for line in proc.stdout:
            print('['+model+'] '+line, end='', flush=True)


def execute(plan, resumed=False):
    import fcntl
    path = Path(plan['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() and not resumed:
            raise FileExistsError('Queue exists; use --resume '+str(path))
        active = {}
        threads = []
        plan.pop('error', None)
        plan.pop('finished', None)
        plan['attempt_peak_tree_rss_bytes']=0
        plan['attempt_started']=dt.datetime.now().isoformat()
        plan.update(status='running', pid=os.getpid(), resumed=resumed)
        for row in plan['records']:
            meta = resolve_run(row['run_dir'])/'training.json'
            if resumed and meta.is_file():
                saved=read(meta)
                row['status'] = 'completed' if saved.get('status') == 'completed' and not saved.get('report_error') else 'pending'
            elif resumed:
                row['status'] = 'pending'
        write(path, plan)
        print('QUEUE: '+str(path)+'\nRESUME: python train.py --resume '+str(path), flush=True)
        def stop(signum, frame):
            raise KeyboardInterrupt('Signal '+str(signum))
        previous = {s: signal.signal(s, stop) for s in [signal.SIGINT, signal.SIGTERM]}
        try:
            while True:
                for pid, (proc, row) in list(active.items()):
                    code = proc.poll()
                    if code is not None:
                        row.update(status='completed' if code == 0 else 'failed', exit_code=code)
                        del active[pid]
                        write(path, plan)
                        if code:
                            raise RuntimeError(row['model']+' failed; see '+row['run_dir'])
                _, available = memory_available()
                rss = process_tree_rss(os.getpid())
                plan['peak_tree_rss_bytes'] = max(plan.get('peak_tree_rss_bytes', 0), rss)
                plan['attempt_peak_tree_rss_bytes']=max(plan['attempt_peak_tree_rss_bytes'],rss)
                if active and (available < plan['reserve_bytes'] or rss > plan['memory_budget_bytes']):
                    raise RuntimeError('Host memory protection triggered; last completed checkpoints remain available. Resume with --jobs 1 and/or a smaller batch in a new run.')
                busy_cards = {r['card'] for _, r in active.values()}
                pending = [r for r in plan['records'] if r['status'] == 'pending']
                for row in pending:
                    if len(active) >= plan['jobs']:
                        break
                    if row['card'] in busy_cards:
                        continue
                    reserved=sum(r.get('memory_reservation_bytes',0) for _,r in active.values())
                    if active and reserved+row.get('memory_reservation_bytes',0)>plan['memory_budget_bytes']:
                        continue
                    # Stagger initialization to avoid two simultaneous checkpoint load peaks.
                    startup_gap = 3 if plan['memory_budget_bytes'] >= plan['jobs']*12*GIB else 30
                    if active and (available < 8*GIB or time.monotonic()-last_start < startup_gap):
                        continue
                    run = Path(row['run_dir'])
                    cmd = row['command']
                    if resumed and (resolve_run(run)/'training.json').exists():
                        cmd = [interpreter(), '-u', str(ROOT/'D.codes/2.utils/competition_cli.py'), 'train', '--resume', str(run)]
                    elif run.exists():
                        raise FileExistsError('Run exists without a resumable training record: '+str(run))
                    env = dict(os.environ, SAR_TRAIN_WORKERS=str(plan['workers']),
                               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               MALLOC_ARENA_MAX='2', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
                    # training_control acquires a physical-card lock in the child.
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, errors='replace', start_new_session=True)
                    row.update(status='running', pid=proc.pid)
                    active[proc.pid] = (proc, row)
                    thread = threading.Thread(target=stream, args=(proc, row['model']), daemon=True)
                    thread.start(); threads.append(thread)
                    busy_cards.add(row['card']); last_start = time.monotonic()
                    write(path, plan)
                if not active and not any(r['status'] == 'pending' for r in plan['records']):
                    break
                time.sleep(1)
            plan['status'] = 'completed'
        except BaseException as error:
            plan.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            for proc, row in active.values():
                proc.terminate()
            for proc, row in active.values():
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                row.update(status='interrupted', exit_code=proc.returncode)
            raise
        finally:
            for thread in threads:
                thread.join(timeout=5)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            plan['finished'] = dt.datetime.now().isoformat()
            plan.setdefault('attempt_history',[]).append(dict(started=plan['attempt_started'],finished=plan['finished'],jobs=plan['jobs'],status=plan['status'],peak_tree_rss_bytes=plan['attempt_peak_tree_rss_bytes']))
            write(path, plan)
            print('Queue '+plan['status']+': '+str(path), flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve the existing named-argument interface for saved commands.
    if argv and argv[0].startswith('-') and argv[0] not in ['--help', '-h', '--resume']:
        from competition_cli import main as legacy
        sys.argv = [sys.argv[0], 'train', *argv]
        return legacy()
    p = parser(); a = p.parse_args(argv)
    if a.resume:
        target = a.resume
        if target == 'last':
            files = sorted(list((ROOT/'.runtime/queues').glob('*.training_queue.json'))+list((ROOT/'C.results/0.settings').glob('*.training_queue.json')), key=lambda f:f.stat().st_mtime)
            if not files:
                p.error('No saved training queue')
            target = str(files[-1])
        target = Path(target).resolve()
        if target.is_dir() or target.name == 'training.json':
            if len(argv)!=2:
                p.error('For a single saved run use only --resume RUN_DIRECTORY')
            from competition_cli import main as legacy
            sys.argv = [sys.argv[0], 'train', '--resume', str(target)]
            return legacy()
        plan = read(target)
        if plan.get('kind') != 'training_queue':
            p.error('Not a training queue JSON')
        if a.dataset or any(flag not in ['--resume', '--jobs', '--memory-gb'] and flag.startswith('--') for flag in argv):
            p.error('Resume keeps the saved configuration; only --jobs and --memory-gb may be changed')
        plan['path'] = str(target)
        if '--jobs' in argv:
            plan['jobs'] = a.jobs
        if '--memory-gb' in argv:
            plan['memory_budget_bytes'] = a.memory_gb*GIB
        if plan['jobs'] > len(plan['cards']):
            p.error('Saved queue has fewer cards than requested jobs')
        preflight(plan)
        return execute(plan, resumed=True)
    if not argv:
        p.print_help(); return
    plan = make_plan(a)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if a.dry_run:
        return
    preflight(plan)
    if a.check:
        return
    execute(plan)


if __name__ == '__main__':
    main()
