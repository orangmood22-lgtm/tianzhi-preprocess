"""Numbered competition workflow. Only explicit commands start NPU work."""
import argparse
import collections
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from obb_labels import load_class_map
from project_io import ROOT, sha, manifest
from training_control import read, write, naming
from experiment_output import resolve_run, public_dir, publish_evaluation


def path(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def prior(value):
    if not isinstance(value, list) or len(value) != 3 or any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in value
    ) or not math.isclose(sum(value), 1., abs_tol=1e-8):
        raise ValueError('请填写 competition.json 的 depth_prior：[一级比例, 二级比例, 三级比例]，非负且和为1')
    return value


def template(dataset):
    return dict(dataset=dataset, depth_prior=None, cards='0,1')


def resolve_config(raw, action):
    """Keep the editable file small; action selects an explicit training recipe."""
    cfg = dict(dataset=raw['dataset'], depth_prior=raw.get('depth_prior'),
                mode='finetune' if action == 'finetune' else 'standard', learning_rate_scale=.1,
                seed=3929, cards='0,1', memory_gb=12, jobs=1,
                detectors={f'model-{i}': dict(epochs=5 if action == 'finetune' else 30, batch=1,
                                             learning_rate=None, obb_epochs=3 if action == 'finetune' else 12,
                                             obb_learning_rate=None) for i in range(1, 6)},
                classifier=dict(epochs=12, batch=8, learning_rate=None, mode='mixed'),
                evaluation=dict(device='0', batch=8, examples=3, postprocess='none',
                                iou=.5, threshold=None, output_depth=None))
    for key in ('seed', 'cards', 'memory_gb', 'jobs', 'learning_rate_scale'):
        if key in raw:
            cfg[key] = raw[key]
    cfg['evaluation']['device'] = str(cfg['cards']).split(',')[0]
    cfg['evaluation'].update(raw.get('evaluation', {}))
    # Old explicit profiles remain readable, but an old mode field never routes a new script.
    for key, profile in raw.get('detectors', {}).items():
        if key not in cfg['detectors']:
            raise ValueError('检测训练仅支持 model-1～5')
        cfg['detectors'][key].update(profile)
    cfg['classifier'].update(raw.get('classifier', {}))
    return cfg


def input_hashes(dataset, candidates):
    data = ROOT / 'A.dataset' / dataset
    files = list(data.glob('*.txt')) + list((data / 'labels').rglob('*.txt')) + list((data / 'annotations').glob('*.json')) + list(data.glob('taxonomy.json')) + list(data.glob('class_map.json'))
    files += [ROOT / 'B.models' / name for name in ('v3_manifest.json', 'sived_manifest.json')]
    files += [ROOT / manifest()['taxonomy']]
    files += [path(cp) for cp in candidates.values()]
    return {str(p): sha(p) for p in files if p.name != 'data.txt'}


def analyze(a):
    from competition_split import load_dataset, main as split_main
    if Path(a.dataset).name != a.dataset or '/' in a.dataset or '\\' in a.dataset or a.dataset in ('.', '..'):
        raise ValueError('数据集必须是 A.dataset 下的文件夹名')
    root = ROOT / 'A.dataset' / a.dataset
    options = argparse.Namespace(class_map=None, group_regex=a.group_regex, label_format='auto')
    if a.ratios:
        split_args = ['--dataset', a.dataset, '--ratios', *map(str, a.ratios), '--seed', str(a.seed)]
        if a.group_regex:
            split_args += ['--group-regex', a.group_regex]
        split_main(split_args)  # Refuses to overwrite existing splits.
    rows, groups, digest, duplicates = load_dataset(root, options)
    by_path = {(root / r['path']).resolve(): i for i, r in enumerate(rows)}
    subsets = {'all': list(range(len(rows)))}
    files = [root / (s + '.txt') for s in ('train', 'val', 'test')]
    if any(p.exists() for p in files):
        if not all(p.exists() for p in files):
            raise ValueError('train/val/test.txt 必须同时存在；未自动补划')
        from yolo_data import read_split
        occupied = set()
        for s in ('train', 'val', 'test'):
            ids = [by_path[p] for p in read_split(root, s)]
            if occupied.intersection(ids):
                raise ValueError('划分重叠')
            occupied.update(ids)
            subsets[s] = ids
        if occupied != set(range(len(rows))):
            raise ValueError('划分没有覆盖全部标注图像')
        membership = {i: s for s in ('train', 'val', 'test') for i in subsets[s]}
        if any(len({membership[i] for i in g['ids']}) > 1 for g in groups):
            raise ValueError('重复图像或同源组跨越划分')
    statistics = {}
    for s, ids in subsets.items():
        objects = [obj for i in ids for obj in rows[i]['objects']]
        depths = collections.Counter(o['depth'] for o in objects)
        statistics[s] = dict(images=len(ids), targets=len(objects), empty=sum(not rows[i]['objects'] for i in ids),
                             depth_counts=[depths[d] for d in (1, 2, 3)], flat_label_targets=depths[0],
                             depth_ratio=[depths[d] / len(objects) for d in (1, 2, 3)] if objects and not depths[0] else None,
                             classes=dict(sorted(collections.Counter(o['label'] for o in objects).items())))
    original_train = None
    if 'train' in subsets and (root / 'labels_original').is_dir():
        from competition_split import parse_line
        class_map = load_class_map(root)
        counts = collections.Counter()
        for index in subsets['train']:
            original = root / 'labels_original' / Path(rows[index]['path']).relative_to('images').with_suffix('.txt')
            if not original.is_file():
                raise ValueError('缺少训练切片的原样标签备份：' + str(original))
            for number, line in enumerate(original.read_text(encoding='utf-8-sig').splitlines(), 1):
                obj = parse_line(line, f'{original}:{number}', class_map)
                if obj:
                    counts[obj['depth']] += 1
        total = sum(counts.values())
        original_train = dict(targets=total, depth_counts=[counts[d] for d in (1, 2, 3)],
                              depth_ratio=[counts[d] / total for d in (1, 2, 3)] if total and not counts[0] else None)
    # Label compatibility is a diagnostic, never a reason to invent a class mapping.
    compatibility = dict(status='unavailable', unknown_labels=[])
    try:
        from competition_metrics import label_action
        tax = read(ROOT / manifest()['taxonomy'])
        unknown = []
        for label in statistics['all']['classes']:
            try:
                label_action(label, tax)
            except ValueError:
                unknown.append(label)
        compatibility = dict(status='incompatible' if unknown else 'compatible', unknown_labels=unknown)
    except (FileNotFoundError, ImportError) as e:
        compatibility['reason'] = str(e)
    config_path = path(a.config) if a.config else ROOT / 'C.results/0.settings' / a.dataset / 'competition.json'
    out = config_path.parent / ('analysis-' + str(time.time_ns()))
    result = dict(dataset=a.dataset, data_sha256=digest, duplicates=duplicates, groups=len(groups),
                  statistics=statistics, classifier_compatibility=compatibility,
                  prior_reference_split='train' if 'train' in statistics else None,
                  original_train_label_depth=original_train)
    write(out / 'analysis.json', result)
    lines = ['集合 | 图像 | 目标 | 空图 | 一级/二级/三级比例', '---|---:|---:|---:|---']
    for s, row in statistics.items():
        lines.append(f"{s} | {row['images']} | {row['targets']} | {row['empty']} | {row['depth_ratio']}")
    lines += ['', '原样训练标签声明深度统计：' + str(original_train),
              '若提供 labels_original/，先验参考原样训练标签的 depth_ratio；否则参考 train 的 depth_ratio。在 competition.json 中手动填写 depth_prior。',
              '无 train 统计时先划分，禁止将 test 分布用于调参。',
              '逐类数量见 analysis.json；分类树检查：' + compatibility['status']]
    (out / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if not config_path.exists():
        cfg = template(a.dataset)
        if a.seed != 3929:
            cfg['seed'] = a.seed
        write(config_path, cfg)
    elif read(config_path)['dataset'] != a.dataset:
        raise ValueError('已有配置属于其他数据集，请指定不同 --config')
    print('\n'.join(lines))
    print(f'统计：{out}\n配置：{config_path}（已有配置不覆盖）')


def run(cmd):
    print('$ ' + shlex.join(list(map(str, cmd))), flush=True)
    proc = subprocess.Popen(list(map(str, cmd)), cwd=ROOT)
    try:
        code = proc.wait()
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def train(a, cfg, state_dir):
    from train_launcher import parser, make_plan
    mode = 'finetune' if a.action == 'finetune' else 'standard'
    if mode not in ('finetune', 'standard'):
        raise ValueError('mode=finetune（默认短程微调）或 standard（原训练配方）；两者均使用现役权重初始化')
    scale = cfg.get('learning_rate_scale', .1) if mode == 'finetune' else 1.
    if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('learning_rate_scale 必须为正数')
    if a.action in ('finetune', 'train-detectors'):
        selected = copy.deepcopy(cfg['detectors'])
    else:
        prior(cfg['depth_prior'])
        # Fail before NPU work when this deployed classifier cannot accept the labels.
        from competition_metrics import label_action
        from competition_split import parse_line
        tax = read(ROOT / manifest()['taxonomy'])
        data = ROOT / 'A.dataset' / cfg['dataset']
        if (data / 'taxonomy.json').exists():
            if read(data / 'taxonomy.json') != tax:
                raise ValueError('分类树与现役权重不同；须先适配输出头和 taxonomy，不能直接微调')
        mapping = load_class_map(data)
        for file in (data / 'labels').rglob('*.txt'):
            for i, line in enumerate(file.read_text(encoding='utf-8-sig').splitlines(), 1):
                obj = parse_line(line, f'{file}:{i}', mapping)
                if obj:
                    label_action(obj['label'], tax)
        selected = {'model-c-s': copy.deepcopy(cfg['classifier'])}
        classifier_mode=selected['model-c-s'].get('mode','mixed')
        if classifier_mode not in ('mixed','level2'):
            raise ValueError('classifier.mode 只能是 mixed 或 level2')
        if classifier_mode=='level2':
            from yolo_data import read_split
            base=(data/'images').resolve()
            depths=[]
            for image in read_split(data,'train'):
                label=data/'labels'/image.relative_to(base).with_suffix('.txt')
                for i,line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(),1):
                    obj=parse_line(line,f'{label}:{i}',mapping)
                    if obj:depths.append(obj['depth'])
            if any(d>2 for d in depths) or 2 not in depths:
                raise ValueError('level2 模式要求 train 标签最高二级且至少有一个二级目标；val 保留原始真值')
    for profile in selected.values():
        if a.epochs:
            profile['epochs'] = a.epochs
        if a.batch:
            profile['batch'] = a.batch
    if not selected or any(key not in {f'model-{i}' for i in range(1, 6)} | {'model-c-s'} for key in selected):
        raise ValueError('训练队列支持 model-1～5（自动配套 OBB）及 model-c-s')
    argv = [cfg['dataset'], ','.join(selected), ','.join(str(v['epochs']) for v in selected.values()),
            '--cards', cfg['cards'], '--memory-gb', str(cfg['memory_gb']), '--jobs', str(cfg['jobs']),
            '--workers', '0', '--seed', str(cfg['seed'])]
    if a.action == 'classifier':
        argv += ['--depth-prior', *map(str, cfg['depth_prior'])]
        if selected['model-c-s'].get('mode','mixed')!='mixed':
            argv += ['--classifier-mode',selected['model-c-s']['mode']]
    plan = make_plan(parser().parse_args(argv))
    initial_hashes = {}
    for row in plan['records']:
        profile = selected[row['model']]
        batch = profile['batch']
        if not isinstance(batch, int) or batch < 1:
            raise ValueError('batch 必须为正整数')
        row['batch_size'] = batch
        cmd = row['command']
        cmd[cmd.index('--batch-size') + 1] = str(batch)
        if row['model'] == 'model-c-s':
            base = read(ROOT / 'D.codes/1.model/modelcs-HierarchicalEnsemble/config.json')['train']
            initial = ROOT / 'B.models/v3_manifest.json'
            row['initialization'] = dict(kind='current_five_branch_classifier', manifest=str(initial), sha256=sha(initial))
            initial_hashes[str(initial)] = sha(initial)
            cmd += ['--checkpoint', str(initial)]
        else:
            registered = read(ROOT / 'B.models/sived_manifest.json')['models'][row['model']]
            base = read(ROOT / registered['config'])['train']
            row['initialization'] = dict(detector=registered['checkpoint'], detector_sha256=registered['sha256'], obb=registered['obb'])
            detector = ROOT / registered['checkpoint']
            geometry = ROOT / registered['obb']['checkpoint']
            initial_hashes[str(detector)] = registered['sha256']
            initial_hashes[str(geometry)] = registered['obb']['sha256']
            cmd += ['--checkpoint', str(detector), '--obb-checkpoint', str(geometry)]
            if mode == 'finetune':
                cmd += ['--finetune']
        lr = profile.get('learning_rate')
        lr = base['learning_rate'] * scale if lr is None else lr
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('learning_rate 必须为正数或 null')
        row.update(training_mode=mode, learning_rate=lr, optimizer_policy='new optimizer; --resume restores saved state')
        cmd += ['--learning-rate', str(lr)]
        if '--obb-epochs' in cmd:
            epochs = profile['obb_epochs']
            if not isinstance(epochs, int) or epochs < 1:
                raise ValueError('obb_epochs 必须为正整数')
            cmd[cmd.index('--obb-epochs') + 1] = str(epochs)
            obb_lr = profile.get('obb_learning_rate')
            obb_lr = .0001 * scale if obb_lr is None else obb_lr
            if not isinstance(obb_lr, (int, float)) or not math.isfinite(obb_lr) or obb_lr <= 0:
                raise ValueError('obb_learning_rate 必须为正数或 null')
            row['obb_learning_rate'] = obb_lr
            cmd += ['--obb-learning-rate', str(obb_lr)]
    if a.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    queue = Path(plan['path'])
    if queue.exists():
        raise FileExistsError(queue)
    for file, digest in initial_hashes.items():
        if sha(Path(file)) != digest:
            raise ValueError('初始化权重已变化：' + file)
    if a.action == 'finetune':
        check_file = state_dir / 'availability.json'
        if not check_file.exists():
            raise ValueError('请先运行第2步：bash 2.check_models.sh ' + cfg['dataset'])
        check = read(check_file)
        if check['status'] != 'completed' or not set(selected) <= set(check['models']):
            raise ValueError('请先通过第2步全部模型可用性检查')
        if check['input_hashes'] != input_hashes(cfg['dataset'], {}):
            raise ValueError('数据/现役资产登记已变化，请重新执行第2步')
    plan['workflow'] = dict(state_dir=str(state_dir), smoke=False, config=cfg,
                           candidate_set='finetune' if a.action == 'finetune' else 'trained' if a.action == 'train-detectors' else 'classifier',
                           initialization_hashes=initial_hashes)
    write(queue, plan)
    write(state_dir / 'last_queue.json', dict(queue=str(queue)))
    print('续训：bash 10.resume.sh --queue ' + shlex.quote(str(queue)), flush=True)
    run([sys.executable, '-u', ROOT / 'train.py', '--resume', queue])
    register(queue)


def register(queue):
    plan = read(queue)
    workflow = plan.get('workflow')
    if not workflow or workflow['smoke'] or plan['status'] != 'completed':
        return
    target = Path(workflow['state_dir']) / ('candidates.' + workflow.get('candidate_set', 'legacy') + '.json')
    candidates = read(target) if target.exists() else {}
    for row in plan['records']:
        cp = resolve_run(row['run_dir']) / row['model'] / 'candidate.json'
        obj = read(cp)
        if obj.get('status') != 'candidate':
            raise ValueError('无有效候选：' + str(cp))
        candidates[row['model']] = str(cp)
    write(target, candidates)
    print('候选索引：' + str(target))


def evaluation(a, cfg, state_dir):
    from evaluation_runtime import selection
    if a.action in ('check', 'validate'):
        if a.action == 'check' and classifier_compatibility(cfg)['status'] != 'compatible':
            cfg = copy.deepcopy(cfg)
            cfg['depth_prior'] = list(manifest()['classifier']['depth_prior'])  # classifier is skipped
        else:
            prior(cfg['depth_prior'])
        models = (selection('all') + selection('all-s') if a.models == 'all-ten' else
                  selection(a.models) if a.models else list(cfg['detectors']))
        selection(','.join(models))
        current = a.action == 'check' or a.current
        if current:
            candidates = {}
        else:
            candidates = read(state_dir / ('candidates.trained.json' if a.trained else 'candidates.finetune.json'))
            candidates.update(read(state_dir / 'candidates.classifier.json'))
        if not current and not set(models + ['model-c-s']) <= set(candidates):
            raise ValueError('候选不齐；先完成检测和分类训练。仅验证现役权重时显式加 --current')
        settings = dict(dataset=cfg['dataset'], models=models, candidates=candidates,
                        depth_prior=cfg['depth_prior'], seed=cfg['seed'], evaluation=cfg['evaluation'])
        settings['input_hashes'] = input_hashes(cfg['dataset'], candidates)
        if a.action == 'check':
            settings['classification_check'] = classifier_compatibility(cfg)
            settings['evaluation'] = dict(settings['evaluation'], postprocess='none')
            print('分类类别兼容性：' + json.dumps(settings['classification_check'], ensure_ascii=False))
    else:
        frozen = read(path(a.frozen) if a.frozen else state_dir / 'frozen.json')
        settings = frozen['settings']
        for p, digest in frozen['hashes'].items():
            if sha(path(p)) != digest:
                raise ValueError('冻结后文件已变化：' + p)
        models = frozen['selected'] if a.action == 'infer' else settings['models']
    out = state_dir / (a.action + '-' + str(time.time_ns()))
    record = dict(status='planned', settings=settings, runs=[], action=a.action)
    e = settings['evaluation']
    devices = str(e.get('devices', e['device'])).split(',')
    if not 1 <= len(devices) <= 8 or len(set(devices)) != len(devices) or any(not v.isdigit() for v in devices):
        raise ValueError('evaluation.devices must contain 1 to 8 distinct numeric NPU IDs')
    groups = [[model] for model in models] if e['postprocess'] == 'none' else [models]
    commands = []
    for group in groups:
        rid = naming(settings['dataset'] + '-' + a.action, group)
        destination = ROOT / 'C.results' / ('3.infer' if a.action == 'infer' else '2.test') / rid
        cmd = [sys.executable, '-u', ROOT / ('infer.py' if a.action == 'infer' else 'test.py')]
        if a.action == 'infer':
            cmd += ['--source', path(a.source), '--models', *group]
        else:
            cmd += [settings['dataset'], ','.join(group), '--split', 'val' if a.action in ('check', 'validate') else 'test']
        class_map_path = path(a.class_map) if a.class_map else ROOT/'A.dataset'/settings['dataset']/'class_map.json'
        if a.class_map or class_map_path.exists():
            load_class_map(path=class_map_path)
            cmd += ['--class-map', class_map_path]
        if a.action == 'check':
            cmd += ['--samples', str(a.samples)]
            if settings['classification_check']['status'] != 'compatible':
                cmd += ['--detection-only']
        cmd += ['--depth-prior', *map(str, settings['depth_prior']), '--batch', str(e['batch']),
                '--device', devices[len(commands) % len(devices)], '--examples', str(e['examples']), '--seed', str(settings['seed']),
                '--postprocess', e['postprocess'], '--iou', str(e['iou']), '--output', destination]
        if e.get('output_depth') is not None:
            if e['output_depth'] not in (1,2,3):raise ValueError('evaluation.output_depth 只能是 1、2、3 或 null')
            cmd += ['--output-depth',str(e['output_depth'])]
        if e['threshold'] is not None:
            cmd += ['--threshold', str(e['threshold'])]
        mapping = {m: settings['candidates'][m] for m in group if m in settings['candidates']}
        if mapping:
            mapfile = out / ('-'.join(group) + '.json')
            if not a.dry_run:
                write(mapfile, mapping)
            cmd += ['--checkpoints', mapfile]
        if 'model-c-s' in settings['candidates']:
            cmd += ['--classifier-checkpoint', settings['candidates']['model-c-s']]
        commands.append(cmd)
        record['runs'].append(str(destination))
    if a.dry_run:
        for cmd in commands:
            print(shlex.join(list(map(str, cmd))))
        return
    write(out / 'evaluation.json', record)
    try:
        failures = run_evaluations(commands)
        record['status'] = 'failed' if failures else 'completed'
        record['failures'] = failures
        summary = []
        for dest in record['runs']:
            if (resolve_run(dest) / 'summary.json').exists():
                summary.extend(read(resolve_run(dest) / 'summary.json'))
        write(out / 'summary.json', summary)
        from evaluation_report import csv_write
        csv_write(out / 'summary.csv', summary)
        if a.action == 'check':
            write(state_dir / 'availability.json', dict(status=record['status'], models=models,
                  input_hashes=settings['input_hashes'], classification=settings['classification_check'],
                  samples=a.samples, evaluation=str(out / 'evaluation.json')))
        if failures:
            raise RuntimeError('部分模型可用性检查失败；已检查其余模型，详情见 ' + str(out))
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        write(out / 'evaluation.json', record)
    print('汇总与报告路径：' + str(out))


def run_evaluations(commands):
    """Independent model groups may use distinct cards; never overlap one card."""
    pending=list(commands);active={};failures=[]
    try:
        while pending or active:
            for card,(proc,cmd) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                del active[card]
                if rc:failures.append(dict(command=list(map(str,cmd)),exit_code=rc))
            for cmd in list(pending):
                card=str(cmd[cmd.index('--device')+1])
                if card in active:continue
                print('$ '+shlex.join(list(map(str,cmd))),flush=True)
                active[card]=(subprocess.Popen(list(map(str,cmd)),cwd=ROOT,start_new_session=True),cmd)
                pending.remove(cmd)
            if active:time.sleep(.2)
    except BaseException:
        for proc,_ in active.values():
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for proc,_ in active.values():
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    return failures


def classifier_compatibility(cfg):
    """Check all val labels before choosing full cascade vs detection-only probing."""
    from competition_split import parse_line
    from competition_metrics import label_action
    from yolo_data import read_split
    data = ROOT / 'A.dataset' / cfg['dataset']
    tax = read(ROOT / manifest()['taxonomy'])
    if (data / 'taxonomy.json').exists() and read(data / 'taxonomy.json') != tax:
        return dict(status='incompatible', reason='dataset taxonomy differs from deployed classifier')
    unknown = set()
    count = 0
    for image in read_split(data, 'val'):
        lab = data / 'labels' / image.relative_to((data / 'images').resolve()).with_suffix('.txt')
        for i, line in enumerate(lab.read_text(encoding='utf-8-sig').splitlines(), 1):
            obj = parse_line(line, f'{lab}:{i}', load_class_map(data))
            if obj:
                count += 1
                try:
                    label_action(obj['label'], tax)
                except ValueError:
                    unknown.add(obj['label'])
    return dict(status='incompatible' if unknown else 'compatible' if count else 'unverified',
                unknown_labels=sorted(unknown), validation_targets=count)


def freeze(a, state_dir):
    from evaluation_runtime import selection
    record_path = path(a.validation)
    if record_path.is_dir():
        record_path /= 'evaluation.json'
    record = read(record_path)
    if record['status'] != 'completed' or record['action'] != 'validate':
        raise ValueError('只能从已完成的完整 val 评测冻结')
    settings = record['settings']
    selected = selection(a.select)
    if not set(selected) <= set(settings['models']):
        raise ValueError('选择了未验证的模型')
    if settings['evaluation']['postprocess'] != 'none' and selected != settings['models']:
        raise ValueError('融合必须冻结实际验证过的完整组合')
    hashes = {**settings['input_hashes'], str(record_path): sha(record_path)}
    for dest in record['runs']:
        dest = resolve_run(dest)
        if read(dest / 'completed.json')['status'] != 'passed':
            raise ValueError('验证未完成')
        request = read(dest / 'request.json')
        if request['dataset']['limited'] or request['dataset']['split'] != 'val':
            raise ValueError('禁止从小样本或 test 冻结')
        hashes[str(dest / 'request.json')] = sha(dest / 'request.json')
        # Resolve exact detector/OBB/classifier/config/taxonomy content recursively.
        def visit(obj):
            if isinstance(obj, dict):
                if 'path' in obj and 'sha256' in obj:
                    hashes[obj['path']] = obj['sha256']
                if 'config' in obj and 'config_sha256' in obj:
                    hashes[obj['config']] = obj['config_sha256']
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, list):
                for value in obj:
                    visit(value)
        visit(request)
    for p, digest in hashes.items():
        if sha(path(p)) != digest:
            raise ValueError('验证后资产已变化：' + p)
    frozen = dict(settings=settings, selected=selected, validation=str(record_path), hashes=hashes)
    dest = state_dir / 'frozen.json'
    if dest.exists():
        raise FileExistsError('已冻结；新方案请使用另一配置目录，避免覆盖 test 的选择依据')
    write(dest, frozen)
    print('已冻结：' + str(dest))


def visualize(a):
    from evaluation_report import report
    source = resolve_run(path(a.run))
    if read(source / 'completed.json')['status'] != 'passed':
        raise ValueError('需要已完成的测试或推理目录')
    request = read(source / 'request.json')
    coco = read(source / 'ground_truth.json')
    meta = request['dataset']
    base = path(meta['source'])
    image_root = base / 'images' if meta['labels_available'] else base.parent if base.is_file() else base
    keys = read(source / 'completed.json')['output_groups']
    predictions = {k: read(source / k / 'predictions.json') for k in keys}
    results = {k: read(source / k / 'metrics.json') for k in keys}
    out = source / ('visualization-' + str(time.time_ns()))
    for key in keys:
        write(out / key / 'predictions.json', predictions[key])
        write(out / key / 'metrics.json', results[key])
    write(out / 'request.json', request)
    report(out, results, predictions, coco, image_root, meta, a.examples, a.seed, request['taxonomy'])
    # Keep the same public contract; new sample images replace the previous sample set.
    from experiment_output import publish_visualization
    publish_visualization(out,public_dir(source))
    print('不重复推理；抽样结果：' + str(public_dir(source) / 'visualization'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['analyze', 'check', 'finetune', 'train-detectors', 'classifier', 'validate', 'test', 'infer', 'visualize', 'status', 'resume'])
    p.add_argument('dataset_name', nargs='?', help='Dataset name; automatically finds its competition.json')
    p.add_argument('--config', help='C.results/0.settings/DATASET/competition.json')
    p.add_argument('--background', action='store_true', help='Detach, record PID and log; Linux only')
    p.add_argument('--dry-run', action='store_true', help='Print resolved commands without training/inference')
    p.add_argument('--dataset')
    p.add_argument('--ratios', nargs=3, type=float)
    p.add_argument('--group-regex')
    p.add_argument('--seed', type=int, default=3929)
    p.add_argument('--models', help='Check/validation models: all / all-s / all-ten / 1,3,5')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch', type=int)
    p.add_argument('--samples', type=int, default=8, help='Step 2: number of validation images per model')
    p.add_argument('--trained', action='store_true', help='Step 5: evaluate optional regular-training candidates instead of fine-tuned ones')
    p.add_argument('--current', action='store_true', help='Validate explicitly selected deployed weights')
    p.add_argument('--validation', help='Completed validation evaluation.json; freeze without rerunning')
    p.add_argument('--select', help='Model IDs to use for unlabeled inference, chosen on val')
    p.add_argument('--frozen')
    p.add_argument('--source')
    p.add_argument('--class-map', type=Path, help='Inference external class -> internal label mapping; defaults to dataset class_map.json')
    p.add_argument('--run', help='Completed individual evaluation directory for visualization')
    p.add_argument('--examples', type=int, default=6)
    p.add_argument('--queue', help='Exact saved training queue path')
    p.add_argument('--job-record', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.class_map and a.action not in ("check", "validate", "test", "infer"):
        p.error("--class-map is for evaluation/inference; for data analysis/training place class_map.json in the dataset directory")
    if a.dataset and a.dataset_name:
        p.error('数据集名称只填写一次')
    a.dataset = a.dataset or a.dataset_name
    if a.samples < 1:
        p.error('--samples 必须大于0')
    if '--samples' in (sys.argv[1:] if argv is None else argv) and a.action != 'check':
        p.error('--samples 仅用于第2步可用性检查；第5/6步使用完整划分')
    if a.dry_run and a.action not in ('check', 'finetune', 'train-detectors', 'classifier', 'validate', 'test', 'infer'):
        p.error('--dry-run 仅支持训练/评测/推理计划')
    if (a.batch is not None or a.epochs is not None) and a.action not in ('finetune', 'train-detectors', 'classifier'):
        p.error('--batch/--epochs 仅用于微调或训练入口')
    if any(v is not None and v < 1 for v in (a.batch, a.epochs)):
        p.error('batch/epochs 必须为正整数')
    if a.models and a.action not in ('check', 'validate') or (a.current or a.trained) and a.action != 'validate':
        p.error('--models 用于第2/5步；--current/--trained 仅用于第5步')
    if a.current and a.trained:
        p.error('--current 与 --trained 不能同时使用')
    if (a.validation or a.select) and (a.dry_run or a.background):
        p.error('冻结操作不使用 --dry-run/--background')
    if a.action == 'analyze':
        if not a.dataset:
            p.error('1.data_analyze.sh --dataset NAME [--ratios .8 .1 .1]')
        return analyze(a)
    if a.action == 'visualize':
        if not a.run or a.examples < 1:
            p.error('--run DIR and positive --examples required')
        return visualize(a)
    if a.action == 'resume':
        if not a.queue:
            p.error('10.resume.sh --queue EXACT_QUEUE.json')
        queue = path(a.queue)
        saved = read(queue)
        if 'workflow' not in saved:
            p.error('Use python train.py --resume for a legacy queue')
        state_dir = Path(saved['workflow']['state_dir'])
        cfg = saved['workflow']['config']
    else:
        if not a.config:
            if not a.dataset:
                p.error('请提供数据集名，例如 bash 3.finetune_all.sh contest')
            a.config = str(ROOT / 'C.results/0.settings' / a.dataset / 'competition.json')
        cfg_path = path(a.config)
        raw = read(cfg_path)
        if a.dataset and raw['dataset'] != a.dataset:
            p.error('数据集与配置不一致')
        cfg = resolve_config(raw, a.action)
        state_dir = cfg_path.parent
    if a.action == 'status':
        availability = state_dir / 'availability.json'
        if availability.exists():
            print('可用性检查：', json.dumps(read(availability), ensure_ascii=False))
        for name in ('finetune', 'classifier', 'trained'):
            candidates = state_dir / ('candidates.' + name + '.json')
            if candidates.exists():
                print('候选 ' + name + ':', str(candidates), ', '.join(read(candidates)))
        for f in sorted(state_dir.glob('job-*.json')):
            print(f.name, json.dumps(read(f), ensure_ascii=False))
        last = state_dir / 'last_queue.json'
        if last.exists():
            q = read(last)['queue']
            plan = read(q)
            print('QUEUE:', q, plan['status'])
            for row in plan['records']:
                print(row['model'], row['status'], row['run_dir'])
        return
    if a.validation or a.select:
        if a.action != 'validate' or not a.validation or not a.select:
            p.error('冻结需同时指定 --validation DIR --select MODELS')
        return freeze(a, state_dir)
    if a.action == 'infer' and not a.source:
        p.error('7.infer.sh requires --source FILE_OR_FOLDER')
    if a.background and not a.dry_run:
        if os.name != 'posix':
            p.error('后台执行仅在 Linux 训练服务器运行')
        job = state_dir / ('job-' + str(time.time_ns()) + '.json')
        log = job.with_suffix('.log')
        command = [sys.executable, '-u', __file__, *[v for v in (sys.argv[1:] if argv is None else argv) if v != '--background'], '--job-record', str(job)]
        write(job, dict(status='starting', command=command, log=str(log)))
        with log.open('a') as stream:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        print(f'后台 PID={proc.pid}\n状态：{job}\n日志：{log}\ntail -f {shlex.quote(str(log))}')
        return
    def stop(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    job = path(a.job_record) if a.job_record else None
    status = dict(status='running', pid=os.getpid(), started=dt.datetime.now().isoformat(), action=a.action)
    if job:
        status = {**read(job), **status}
        write(job, status)
    try:
        # Serialize operations that update candidate state for the same configuration.
        import fcntl
        with (state_dir / 'workflow.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.action in ('finetune', 'train-detectors', 'classifier'):
                train(a, cfg, state_dir)
            elif a.action == 'resume':
                run([sys.executable, '-u', ROOT / 'train.py', '--resume', queue])
                register(queue)
            else:
                if a.action == 'check' and not a.dry_run:
                    write(state_dir / 'availability.json', dict(status='running', models=[]))
                evaluation(a, cfg, state_dir)
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        if a.action == 'check' and not a.dry_run:
            availability = state_dir / 'availability.json'
            record = read(availability) if availability.exists() else dict(models=[])
            record.update(status=status['status'], error=str(exc))
            write(availability, record)
        raise
    finally:
        if job:
            status['finished'] = dt.datetime.now().isoformat()
            write(job, status)


if __name__ == '__main__':
    main()
