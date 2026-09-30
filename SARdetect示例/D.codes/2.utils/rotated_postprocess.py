"""Optional rotated NMS and WBF. No model or project-directory side effects.

Coordinates are image-pixel quadrilaterals. WBF fits rectangles, aligns equivalent
width/height representations to the seed, averages doubled angles, and regenerates
a rectangle. Model weights are positive; a cluster accepts one box per model.
"""
import math
from collections import defaultdict
import cv2
import numpy as np


def quad(points):
    p = np.asarray(points, np.float64)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError('Expected four finite pixel-coordinate pairs')
    center = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1]-center[1], p[:, 0]-center[0]))]
    e = np.roll(p, -1, axis=0)-p
    if np.any(e[:, 0]*np.roll(e[:, 1], -1)-e[:, 1]*np.roll(e[:, 0], -1) <= 1e-8):
        raise ValueError('Degenerate or non-convex box')
    return p.astype(np.float32)


def bounds(p):
    return np.r_[p.min(0), p.max(0)]


def iou(a, b):
    a, b = quad(a), quad(b)
    return fast_iou(a, b, bounds(a), bounds(b))


def fast_iou(a, b, ab, bb):
    if ab[2] <= bb[0] or bb[2] <= ab[0] or ab[3] <= bb[1] or bb[3] <= ab[1]:
        return 0.
    area, _ = cv2.intersectConvexConvex(a, b)
    return float(np.clip(area/max(cv2.contourArea(a)+cv2.contourArea(b)-area, 1e-9), 0, 1))


def normalize(rows, model):
    result = []
    for index, row in enumerate(rows):
        score = float(row['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid confidence')
        points = quad(row['quad'])
        # Explicitly discard old classification probabilities: fused crops need inference.
        keep = {k: row[k] for k in ('image_id', 'score', 'bbox', 'quad')}
        result.append(dict(keep, model=str(model), prediction_id=f'{model}:{index}',
                           category_id=0, quad=points.tolist()))
    return result


def nms(rows, threshold=.5, class_agnostic=True, prefer_complete=False):
    if not 0 <= threshold <= 1:
        raise ValueError('IoU threshold outside [0,1]')
    ordered = sorted(rows, key=lambda r: (bool(r.get('edge', False)) if prefer_complete else False,
                                          -r['score'], r['prediction_id']))
    kept, geometries = [], []
    for r in ordered:
        p = quad(r['quad']); b = bounds(p)
        if any((class_agnostic or r.get('category_id', 0) == q.get('category_id', 0))
               and fast_iou(p, op, b, ob) > threshold
               for q, (op, ob) in zip(kept, geometries)):
            continue
        kept.append(r); geometries.append((p, b))
    return sorted(kept, key=lambda r: (-r['score'], r['prediction_id']))


def rectangle(p, reference=None):
    (x, y), (w, h), angle = cv2.minAreaRect(quad(p))
    angle = math.radians(angle)
    if reference is None:
        if h > w:
            w, h, angle = h, w, angle+math.pi/2
    else:
        # Same physical rectangle has (w,h,theta) and (h,w,theta+pi/2).
        distance = lambda a: abs(math.atan2(math.sin(2*(a-reference)), math.cos(2*(a-reference))))
        if distance(angle+math.pi/2) < distance(angle):
            w, h, angle = h, w, angle+math.pi/2
    return np.array([x, y, w, h, angle], np.float64)


def fuse(cluster, weights, score_mode):
    anchor = rectangle(cluster[0]['quad'])
    rr = np.stack([rectangle(r['quad'], anchor[4]) for r in cluster])
    ww = np.array([weights[r['model']]*r['score'] for r in cluster])
    geometry_weights = ww if ww.sum() > 0 else np.array([weights[r['model']] for r in cluster])
    center_size = np.average(rr[:, :4], axis=0, weights=geometry_weights)
    cs = np.average(np.c_[np.cos(2*rr[:, 4]), np.sin(2*rr[:, 4])], axis=0, weights=geometry_weights)
    angle = math.atan2(cs[1], cs[0])/2 if np.linalg.norm(cs) > 1e-6 else anchor[4]
    p = quad(cv2.boxPoints((tuple(center_size[:2]), tuple(center_size[2:]), math.degrees(angle))))
    bb = bounds(p)
    denominator = sum(weights.values()) if score_mode == 'support' else sum(weights[r['model']] for r in cluster)
    return dict(image_id=cluster[0]['image_id'], score=float(ww.sum()/denominator),
                bbox=[float(bb[0]), float(bb[1]), float(bb[2]-bb[0]), float(bb[3]-bb[1])],
                quad=p.tolist(), category_id=0, prediction_id='wbf:'+','.join(r['prediction_id'] for r in cluster),
                sources=[r['prediction_id'] for r in cluster], models=sorted(r['model'] for r in cluster),
                support=len(cluster))


def wbf(rows, weights, threshold=.5, score_mode='support'):
    if score_mode not in ('support', 'present') or not 0 <= threshold <= 1:
        raise ValueError('Invalid WBF configuration')
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError('Positive finite model weights required')
    clusters, reps, models = [], [], []
    for row in sorted(rows, key=lambda r: (-r['score']*weights[r['model']], r['prediction_id'])):
        q = quad(row['quad']); qb = bounds(q)
        overlaps = [(fast_iou(q, p, qb, b), i) for i, (p, b) in enumerate(reps) if row['model'] not in models[i]]
        overlap, ident = max(overlaps, default=(0., -1))
        if overlap > threshold:
            clusters[ident].append(row); models[ident].add(row['model'])
            p = quad(fuse(clusters[ident], weights, score_mode)['quad']); reps[ident] = (p, bounds(p))
        else:
            clusters.append([row]); models.append({row['model']}); reps.append((q, qb))
    return sorted([fuse(c, weights, score_mode) for c in clusters], key=lambda r: (-r['score'], r['prediction_id']))


def process(model_rows, config):
    groups = defaultdict(list)
    weights = {str(m): float(config.get('weights', {}).get(str(m), 1.)) for m in config['models']}
    for m in config['models']:
        for r in model_rows[str(m)]:
            if r['score'] >= config.get('input_floor', .05):
                groups[r['image_id']].append(r)
    result = []
    for ident in sorted(groups):
        rows = groups[ident]
        if config['method'] == 'nms':
            rows = nms(rows, config['iou'])
        elif config['method'] in ('wbf', 'wbf_nms'):
            rows = wbf(rows, weights, config['iou'], config['score_mode'])
            if config['method'] == 'wbf_nms':
                rows = nms(rows, config.get('final_nms', .5))
        elif config['method'] != 'none':
            raise ValueError('Unknown postprocessor')
        result.extend(rows)
    return result
