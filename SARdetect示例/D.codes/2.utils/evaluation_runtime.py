"""Uniform detector -> learned OBB -> optional fusion -> hierarchical classification."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from project_io import ROOT, UTIL, read, write, sha
from experiment_output import work_dir, initialize, publish_evaluation


def selection(value):
    if value in ('all', 'all-s'):
        return [f'model-{i}'+('-s' if value=='all-s' else '') for i in range(1,6)]
    values = value.split(',')
    names = ['model-'+x if not x.startswith('model-') else x for x in values]
    allowed = {f'model-{i}{s}' for i in range(1,6) for s in ('','-s')}
    if not names or len(set(names)) != len(names) or not set(names) <= allowed:
        raise argparse.ArgumentTypeError('Use 1..5, 1-s..5-s, comma-separated IDs, all, or all-s')
    return names


def parser(action):
    p = argparse.ArgumentParser(description='Unified OBB detection + hierarchical classification; current Python environment')
    p.add_argument('dataset_name', nargs='?', help='Folder in A.dataset')
    p.add_argument('model_list', nargs='?', help='1 | 1,3,5 | all | 1-s,5-s | all-s')
    p.add_argument('--class-map', type=Path, help='External class token -> model label JSON; auto-loads dataset class_map.json')
    p.add_argument('--dataset'); p.add_argument('--model'); p.add_argument('--models', nargs='+')
    p.add_argument('--source', type=Path, help='Inference only: image file or folder; no scores without labels')
    p.add_argument('--split', choices=['val','test'], default='test')
    p.add_argument('--device', default='0', help='One physical NPU ID, e.g. 0 or npu:1')
    p.add_argument('--checkpoint', type=Path, help='One detector training candidate.json or raw weight')
    p.add_argument('--checkpoints', type=Path, help='JSON map: {"model-1":"C.results/.../candidate.json", ...}')
    p.add_argument('--obb-checkpoint', type=Path, help='Optional geometry checkpoint/candidate for one detector')
    p.add_argument('--classifier-checkpoint', type=Path, help='Classifier candidate.json / candidate_manifest.json')
    p.add_argument('--depth-prior', nargs=3, type=float, metavar=('P1','P2','P3'))
    p.add_argument('--output-depth', type=int, choices=[1,2,3], help='Force prediction labels to one depth; default uses classifier candidate policy')
    p.add_argument('--postprocess', choices=['none','nms','wbf','wbf-nms'], default='none',
                   help='none: independent models; otherwise combine selected models into one ensemble')
    p.add_argument('--iou', type=float, default=.5, help='NMS suppression or WBF cluster IoU')
    p.add_argument('--final-nms-iou', type=float, default=.5)
    p.add_argument('--score-mode', choices=['support','present'], default='support')
    p.add_argument('--weights', help='Positive comma-separated WBF weights in model order; default equal')
    p.add_argument('--threshold', '--score-threshold', dest='threshold', type=float,
                   help='Final confidence; independent models default to registered thresholds; ensemble default 0.5')
    p.add_argument('--input-floor', type=float, default=.05, help='Minimum confidence before OBB regression/fusion')
    p.add_argument('--batch', type=int, default=16, help='OBB/classifier crop batch, default 16')
    p.add_argument('--samples', type=int, help='Functional check only: evaluate first N images; default complete split')
    p.add_argument('--examples', type=int, default=3, help='Random samples per high/low score group; 0 disables pictures')
    p.add_argument('--seed', type=int, default=3931)
    p.add_argument('--output', type=Path); p.add_argument('--run-id')
    p.add_argument('--dry-run', action='store_true', help='Validate data/assets and write configuration without model inference')
    p.add_argument('--detection-only', action='store_true', help='Check detection/OBB without classifier; allows a different label taxonomy')
    return p


def run_backend(key, asset, coco, image_root, dest, split):
    from training_resources import interpreter
    raw = read(asset['config']); kind = raw['backend']
    raw['model']['pretrained_detector'] = asset['detector']['path']
    raw['run'] = dict(id='detector', output_root=str(dest/'raw'))
    raw['train']['workers'] = 0
    raw['data'].update(root=str(dest/'annotations'), image_prefix=str(image_root)+'/',
                       **{f'{s}_annotations':f'{s}.json' for s in ('train','val','test')})
    for s in ('train','val','test'):
        write(dest/f'annotations/{s}.json', coco if s==split else dict(images=[],annotations=[],categories=coco['categories']))
    config = dest/'resolved.json'; write(config, raw)
    script = dict(faster='faster_evaluate.py',openmmlab='openmmlab_backend.py',rtdetr='rtdetr_backend.py',yolo='yolo_backend.py')[kind]
    cmd = [interpreter(kind=='yolo'),str(UTIL/script),'--config',str(config),'--checkpoint',asset['detector']['path'],
           '--run-id','detector','--split',split,'--dump-predictions']
    cmd += ['--output-root',str(dest/'raw')] if kind=='faster' else ['--mode','test']
    if kind in ('rtdetr','yolo'):
        cmd += ['--device','npu:0']
    write(dest/'command.json', dict(command=cmd, started=time.time()))
    with (dest/'detector.log').open('w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        write(dest/'process.json', dict(pid=process.pid, command=cmd))
        start = time.monotonic()
        while True:
            try:
                code = process.wait(timeout=30); break
            except subprocess.TimeoutExpired:
                print(f'  {key}: detector running, {time.monotonic()-start:.0f}s; log: {dest}/detector.log', flush=True)
            except BaseException:
                process.terminate(); process.wait(); raise
    if code:
        raise RuntimeError(f'{key} detector failed (exit {code}); see {dest}/detector.log')
    from result_artifacts import _coco_predictions
    rows = _coco_predictions(dest/'raw/detector', coco)
    write(dest/'hbb_predictions.json', rows)
    return rows


def classify(rows, coco, image_root, cf, tax, batch):
    import torch, torch_npu
    import numpy as np
    from model_loader import load_model_module
    from competition_metrics import decode, actions, action_label
    torch.npu.set_device(0); torch.set_num_threads(2)
    ensemble = load_model_module('model-c-s','ensemble').ClassifierEnsemble(ROOT,tax,cf['assets'])
    q = ensemble.predict(rows,coco,image_root,batch_size=batch)
    indices,_ = decode(q,tax,cf['margin'],cf['depth_prior'],cf.get('output_depth')); aa = actions(tax)
    output = []
    for r, probs, i in zip(rows, q, indices):
        depth, node = aa[i]
        output.append(dict(r, probabilities=probs.tolist(), label=action_label(aa[i],tax), label_depth=depth,
                           label_ids=list(node), category_id=node[depth-1], category_level=depth, leaf_top1=int(np.argmax(probs))))
    del ensemble; torch.npu.empty_cache()
    return output


def main(action='test', argv=None):
    p = parser(action); a = p.parse_args(argv)
    if a.dataset and a.dataset_name or sum(bool(x) for x in (a.model_list,a.model,a.models))>1:
        p.error('Use positional arguments or --dataset/--models, without duplicates')
    data = a.dataset or a.dataset_name
    try:
        models = selection(','.join(a.models) if a.models else a.model_list or a.model or '1')
    except argparse.ArgumentTypeError as e:
        p.error(str(e))
    if a.source and (action!='infer' or data) or not a.source and not data:
        p.error('Specify a dataset, or use infer.py --source FILE_OR_FOLDER --model 1')
    device = a.device.removeprefix('npu:')
    if not device.isdigit():
        p.error('--device selects one physical NPU')
    for name in ('iou','final_nms_iou','input_floor','threshold'):
        value = getattr(a,name)
        if value is not None and (not math.isfinite(value) or not 0<=value<=1):
            p.error(name+' must be finite and within [0,1]')
    if a.batch<1 or a.examples<0 or a.samples is not None and a.samples<1:
        p.error('Positive batch/samples and non-negative examples required')
    if a.weights and not a.postprocess.startswith('wbf'):
        p.error('--weights requires WBF')
    weights = [1.]*len(models)
    if a.weights:
        try: weights = list(map(float,a.weights.split(',')))
        except ValueError: p.error('Invalid --weights')
        if len(weights)!=len(models) or any(not math.isfinite(w) or w<=0 for w in weights):
            p.error('One positive finite weight per selected model required')
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device
    import fcntl
    card_lock=ROOT/'.runtime/locks'/('npu-'+device+'.lock')
    card_lock.parent.mkdir(parents=True,exist_ok=True)
    card_handle=card_lock.open('a')
    try:fcntl.flock(card_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('NPU '+device+' is in use by another managed job')
    import signal
    def stop(signum,frame):raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM,stop)
    from evaluation_assets import resolve
    assets, cf, tax = resolve(models,a.checkpoint,a.checkpoints,a.obb_checkpoint,a.classifier_checkpoint,a.depth_prior,detection_only=a.detection_only)
    if a.output_depth is not None:cf['output_depth']=a.output_depth
    from evaluation_data import load
    coco, image_root, meta = load(data,a.split,tax,a.source,a.samples,detection_only=a.detection_only,class_map=a.class_map)
    meta['detection_only'] = a.detection_only
    from training_control import naming, valid_name
    rid = a.run_id or naming(a.source.stem if a.source else data, models)
    if not valid_name(rid):
        p.error('Run name: <dataset>_modelN_cfg00_YYYYMMDDHHMMSS')
    out = (a.output or ROOT/'C.results'/('3.infer' if action=='infer' else '2.test')/rid).resolve()
    if out.is_relative_to(ROOT/'C.results') and not valid_name(out.name):
        p.error('C.results directory must follow the standard run naming')
    if out.exists():
        raise FileExistsError('Refusing to overwrite: '+str(out))
    public=out
    out=work_dir(public)
    if out.exists():raise FileExistsError('Internal experiment already exists: '+str(out))
    thresholds = {k:float(a.threshold if a.threshold is not None else v['threshold']) for k,v in assets.items()}
    final_threshold = float(.5 if a.threshold is None else a.threshold)
    if a.input_floor > (min(thresholds.values()) if a.postprocess=='none' else final_threshold):
        p.error('--input-floor must not exceed final threshold')
    config = dict(action=action, arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  models=models, assets=assets, classifier=cf, taxonomy=tax, dataset=meta, thresholds=thresholds,
                  postprocess=dict(method=a.postprocess.replace('-','_'), models=models, weights=dict(zip(models,weights)),
                                   input_floor=a.input_floor, iou=a.iou, final_nms=a.final_nms_iou, score_mode=a.score_mode),
                  ensemble_threshold=final_threshold, note='External OBB postprocessing defaults OFF; native detector filtering/NMS remains.',
                  scoring='proxy-v3.2; polygon IoU>=0.3; per-image macro; real declared GT depths; empty correct=1',
                  python=sys.executable, pid=os.getpid(), started=time.time())
    initialize(public,config,'运行中')
    write(out/'request.json',config); write(out/'ground_truth.json',coco)
    config['annotation_sha256'] = sha(out/'ground_truth.json'); write(out/'request.json',config)
    print(f"Models: {', '.join(models)} -> learned OBB -> {'detection only' if a.detection_only else 'model-c-s hierarchical ensemble'}\n"
          f"Data: {meta['images']} images / {meta['targets']} GT targets; split={meta['split'] or 'unlabeled'}; limited={meta['limited']}\n"
          f"Postprocess: {a.postprocess}; classifier depth prior={cf['depth_prior']}; NPU={device}; crop batch={a.batch}\n"
          f"Output: {public}", flush=True)
    if a.dry_run:
        write(out/'completed.json',dict(status='dry-run')); publish_evaluation(out,public,status='dry-run'); return
    try:
        previous_cwd=Path.cwd()
        os.chdir(out)  # Backend/compiler scratch output belongs to internal run state.
        execute(a, models, assets, cf, tax, coco, image_root, meta, thresholds, config, out)
    except BaseException:
        write(out/'failed.json',dict(status='failed',error=traceback.format_exc(),finished=time.time()))
        publish_evaluation(out,public,status='failed')
        raise
    finally:
        os.chdir(previous_cwd)
    publish_evaluation(out,public,status='passed')


def execute(a, models, assets, cf, tax, coco, image_root, meta, thresholds, config, out):
    from rotated_postprocess import normalize, process
    from training_resources import release_host_memory
    raw = {}; start = time.monotonic()
    for i, key in enumerate(models,1):
        print(f'[{i}/{len(models)}] {key}: detecting and predicting OBB', flush=True)
        hbb = run_backend(key,assets[key],coco,image_root,out/key,a.split)
        from sived_obb import refine
        rows = refine(coco,hbb,image_root,assets[key]['obb']['path'],floor=a.input_floor,batch=a.batch)
        raw[key] = normalize(rows,key); write(out/key/'obb_predictions.json',raw[key])
        import torch
        torch.npu.empty_cache(); release_host_memory()
        print(f'  {len(rows)} OBB proposals; elapsed {time.monotonic()-start:.0f}s; remaining detector estimate '
              f'{(time.monotonic()-start)/i*(len(models)-i):.0f}s', flush=True)
    if a.postprocess=='none':
        groups = {k:[r for r in raw[k] if r['score']>=thresholds[k]] for k in models}
    else:
        groups = {'ensemble-'+a.postprocess:[r for r in process(raw,config['postprocess']) if r['score']>=config['ensemble_threshold']]}
    # One classifier call for all independent outputs, sequential branch loading.
    keys = list(groups); sizes = [len(groups[k]) for k in keys]
    combined = [r for k in keys for r in groups[k]]
    print('Classifier skipped (detection-only check)' if a.detection_only else
          f'Classifying {len(combined)} final boxes ({len(keys)} output group(s)); fused boxes are recropped',flush=True)
    # Heartbeats keep long classifier runs observable without backend console noise.
    import threading
    stop = threading.Event(); before = time.monotonic(); terminal = sys.stdout
    def heartbeat():
        while not stop.wait(30):
            print(f'  Classifier running: {time.monotonic()-before:.0f}s',file=terminal,flush=True)
    thread = threading.Thread(target=heartbeat,daemon=True); thread.start()
    from contextlib import redirect_stdout, redirect_stderr
    try:
        with (out/'classifier.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            if not a.detection_only:
                combined = classify(combined,coco,image_root,cf,tax,a.batch)
    finally: stop.set(); thread.join()
    offset = 0; results = {}; predictions = {}
    from evaluation_report import evaluate, report
    for key,size in zip(keys,sizes):
        rows = combined[offset:offset+size]; offset += size; predictions[key] = rows
        result = evaluate(coco,rows,tax,cf,meta)
        results[key] = result
        write(out/key/'predictions.json',rows); write(out/key/'metrics.json',result)
        from obb_labels import export_predictions
        # Detection-only diagnostics have no classified category; label them vehicle.
        export_map = {} if a.detection_only else meta.get('class_map', {})
        receipt = export_predictions(coco,rows,out/key/'labels',export_map)
        write(out/key/'labels-format.json',receipt)
    report(out,results,predictions,coco,image_root,meta,a.examples,a.seed,tax)
    write(out/'completed.json',dict(status='passed',finished=time.time(),seconds=time.monotonic()-start,
                                   output_groups=keys,models=models,postprocess=a.postprocess))
    print('Model/group                  Total      Det      Cls       FN       FP',flush=True)
    for key,r in results.items():
        f = lambda k: 'N/A' if r.get(k) is None else f"{r[k]*100:.2f}%"
        print(f"{key:28s} {f('joint_score'):>8s} {f('det_score'):>8s} {f('class_score'):>8s} {str(r.get('fn','N/A')):>8s} {str(r.get('fp','N/A')):>8s}")
    from experiment_output import public_dir
    print(f'Result: {public_dir(out)}/result.txt',flush=True)
