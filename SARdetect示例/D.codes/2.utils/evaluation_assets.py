"""Resolve detector, geometry and classifier independently, including training candidates."""
import copy
from pathlib import Path
from project_io import ROOT, read, sha, manifest


def checked_path(value, digest=None):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT/p
    p = p.resolve(strict=True)
    actual = sha(p)
    if digest and actual != digest:
        raise ValueError('Asset hash mismatch: '+str(p))
    return dict(path=str(p), sha256=actual)


def candidate(path, model=None):
    asset = checked_path(path); p = Path(asset['path'])
    if p.suffix != '.json':
        return dict(checkpoints=[asset])
    row = read(p)
    if row.get('status') in ('failed', 'invalid'):
        raise ValueError('Candidate failed validation: '+str(p))
    if model and row.get('model') and row['model'] != model:
        raise ValueError('Candidate model mismatch: '+str(row['model'])+' vs '+model)
    return row


def resolve(models, checkpoint=None, checkpoints=None, obb_checkpoint=None, classifier=None, prior=None, detection_only=False):
    from competition_metrics import validate_depth_prior
    v3 = manifest(); sived = read(ROOT/'B.models/sived_manifest.json')
    cfg = copy.deepcopy(v3)
    if classifier:
        cfg = candidate(classifier, 'model-c-s')
        if 'manifest' in cfg:
            asset = checked_path(cfg['manifest'], cfg.get('sha256')); cfg = read(asset['path'])
        if cfg.get('classifier', {}).get('kind') != 'leaf-ensemble-knn':
            raise ValueError('Use the classifier training candidate.json or candidate_manifest.json')
    tax = read(ROOT/cfg['taxonomy'])
    cf = copy.deepcopy(cfg['classifier'])
    cf['depth_prior'] = list(validate_depth_prior(prior if prior is not None else cf['depth_prior']))
    # Validate all classifier assets before spending time on detection.
    for name, row in ({} if detection_only else cf['assets']).items():
        if name == 'legacy':
            for member in row.values():
                checked_path(member['path'], member['sha256'])
        elif isinstance(row, dict) and 'path' in row:
            checked_path(row['path'], row['sha256'])
    overrides = read(checkpoints) if checkpoints else {}
    if not isinstance(overrides, dict) or set(overrides)-set(models):
        raise ValueError('--checkpoints requires a JSON map of selected model IDs to candidate paths')
    if checkpoint:
        if len(models) != 1 or overrides:
            raise ValueError('--checkpoint requires one model and no --checkpoints')
        overrides[models[0]] = str(checkpoint)
    result = {}
    for key in models:
        row = copy.deepcopy(v3['detectors'][key] if key.endswith('-s') else sived['models'][key])
        obb = row.get('obb', copy.deepcopy(v3['refiner']))
        cp = dict(path=row['checkpoint'], sha256=row['sha256'])
        if key in overrides:
            c = candidate(overrides[key], key)
            if not c.get('checkpoints'):
                raise ValueError('Detector candidate has no checkpoints')
            cp = c['checkpoints'][0]
            obb = c.get('obb') or obb
            row['threshold'] = c.get('threshold', row['threshold'])
        if obb_checkpoint:
            if len(models) != 1:
                raise ValueError('--obb-checkpoint requires one detector')
            c = candidate(obb_checkpoint)
            obb = c.get('obb') or c.get('branches', {}).get('refiner') or c.get('checkpoints', [None])[0]
            if not obb:
                raise ValueError('No refiner checkpoint in candidate')
        cp = checked_path(cp['path'], cp.get('sha256'))
        obb = checked_path(obb.get('checkpoint', obb.get('path')), obb.get('sha256'))
        result[key] = dict(detector=cp, obb=obb, config=str(ROOT/row['config']),
                           config_sha256=sha(ROOT/row['config']), threshold=row['threshold'])
    return result, cf, tax
