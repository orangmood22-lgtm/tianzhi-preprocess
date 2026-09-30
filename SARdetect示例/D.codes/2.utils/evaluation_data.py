"""Read one evaluation split; preserve real label depth and original polygons."""
from pathlib import Path
from PIL import Image
from project_io import ROOT, read


def load(value, split, tax, source=None, limit=None, detection_only=False, class_map=None):
    from competition_metrics import label_action
    from competition_split import parse_line, IMAGE_EXTS
    from obb_labels import load_class_map
    mapping = load_class_map(path=class_map) if class_map else {}
    from rotated_postprocess import quad
    if source:
        p = Path(source).resolve(strict=True)
        base = p.parent if p.is_file() else p
        files = [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.suffix.lower() in IMAGE_EXTS and x.is_file())
        c = dict(images=[], annotations=[], categories=[dict(id=1, name='vehicle')])
        for i, f in enumerate(files, 1):
            with Image.open(f) as im:
                w, h = im.size
            c['images'].append(dict(id=i, file_name=f.relative_to(base).as_posix(), width=w, height=h))
        labelled, hierarchical = False, False
    else:
        p = Path(value)
        p = p if p.is_absolute() else ROOT/'A.dataset'/p
        p = p.resolve(strict=True); base = (p/'images').resolve(strict=True)
        labelled = True
        if not class_map: mapping = load_class_map(p)
        if (p/'taxonomy.json').exists() and not detection_only and read(p/'taxonomy.json') != tax:
            raise ValueError('Dataset taxonomy differs from classifier taxonomy; supply compatible classifier weights')
        if not (p/'labels').is_dir() and (p/f'annotations/{split}.json').is_file():
            c = read(p/f'annotations/{split}.json')
            hierarchical = not detection_only
        else:
            from yolo_data import read_split, split_overlap_audit
            if any(split_overlap_audit(p).values()):
                raise ValueError('Dataset splits overlap')
            c = dict(images=[], annotations=[], categories=[dict(id=1, name='vehicle')]); kinds = set()
            for i, f in enumerate(read_split(p, split), 1):
                name = f.relative_to(base)
                with Image.open(f) as im:
                    w, h = im.size
                c['images'].append(dict(id=i, file_name=name.as_posix(), width=w, height=h))
                lab = p/'labels'/name.with_suffix('.txt')
                for n, line in enumerate(lab.read_text(encoding='utf-8-sig').splitlines(), 1):
                    obj = parse_line(line, f'{lab}:{n}', mapping)
                    if not obj:
                        continue
                    xy = obj['coordinates']; q = [xy[j:j+2] for j in range(0, 8, 2)]
                    x, y = max(0, min(xy[::2])), max(0, min(xy[1::2]))
                    bw, bh = min(w, max(xy[::2]))-x, min(h, max(xy[1::2]))-y
                    if min(bw, bh) <= 0:
                        raise ValueError(f'Box outside image: {lab}:{n}')
                    r = dict(id=len(c['annotations'])+1, image_id=i, category_id=1, bbox=[x,y,bw,bh],
                             quad=q, area=bw*bh, iscrowd=0, difficult=obj['difficult'])
                    # Explicit native SIVED sentinel; never treat unknown competition labels as generic.
                    try:
                        if detection_only:
                            kinds.add('generic')
                            c['annotations'].append(r)
                            continue
                        d, node = label_action(obj['label'], tax)
                    except ValueError:
                        if obj['label'] != 'vehicle-vehicle-vehicle':
                            raise
                        kinds.add('generic')
                    else:
                        kinds.add('hierarchical')
                        r.update(l1=node[0], l2=node[1], l3=node[2], mixed_label_depth=d)
                    c['annotations'].append(r)
            if len(kinds) > 1:
                raise ValueError('Do not mix generic vehicle and hierarchical ground truth in one evaluation split')
            hierarchical = kinds != {'generic'}
    if detection_only:
        hierarchical = False
        for r in c['annotations']:
            for field in ('mixed_label_depth', 'l1', 'l2', 'l3'):
                r.pop(field, None)
    if not c['images']:
        raise ValueError('The selected split/source contains no images')
    ids = {im['id'] for im in c['images']}
    if len(ids) != len(c['images']):
        raise ValueError('Duplicate image IDs')
    for im in c['images']:
        f = (base/im['file_name']).resolve(strict=True)
        if not f.is_relative_to(base):
            raise ValueError('Image path escapes images directory')
    for r in c['annotations']:
        if r['image_id'] not in ids:
            raise ValueError('Annotation references unknown image')
        r['quad'] = quad(r['quad']).tolist()
        if hierarchical:
            d = r['mixed_label_depth']; node = (r['l1'], r['l2'], r['l3'])
            from competition_metrics import actions
            active = [(depth, n[:depth]) for depth, n in actions(tax)]
            if (d, node[:d]) not in active:
                raise ValueError('Ground-truth hierarchy path absent from classifier taxonomy')
    total = len(c['images'])
    if limit:
        c['images'] = c['images'][:limit]; ids = {im['id'] for im in c['images']}
        c['annotations'] = [r for r in c['annotations'] if r['image_id'] in ids]
    return c, base, dict(source=str(p), labels_available=labelled, classification_labels=hierarchical,
                        split=split if not source else None, available_images=total, class_map=mapping,
                        images=len(c['images']), targets=len(c['annotations']), limited=bool(limit and limit<total))
