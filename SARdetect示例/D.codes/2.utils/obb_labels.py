"""Competition wire format: class + eight pixel coordinates [+ confidence].

Class tokens are strings, including numeric IDs. Hierarchy mapping is optional
and explicit; an external class ID is never inferred from a model head index.
"""
import json
import math
import re
from pathlib import Path


def tokens(line):
    try:
        decoder = json.JSONDecoder()
        result = []
        text = line.strip()
        while text:
            if text.startswith('"'):
                value, end = decoder.raw_decode(text)
                if not isinstance(value, str) or (text[end:] and not text[end].isspace()):
                    raise ValueError('Invalid quoted class')
                result.append(value)
                text = text[end:].lstrip()
            else:
                value, *tail = text.split(maxsplit=1)
                result.append(value)
                text = tail[0] if tail else ''
        return result
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid label tokens: ' + line) from exc


def class_token(value):
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError('Class must be a name or integer ID')
    value = str(value)
    if not value or value != value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError('Empty class or control characters in class')
    return value


def coordinates(values):
    xy = [float(v) for v in values]
    if len(xy) != 8 or not all(math.isfinite(v) for v in xy):
        raise ValueError('Expected eight finite pixel coordinates')
    points = list(zip(xy[::2], xy[1::2]))
    cross = []
    for i in range(4):
        a, b, c = points[i], points[(i+1) % 4], points[(i+2) % 4]
        cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
    if not (all(v > 0 for v in cross) or all(v < 0 for v in cross)):
        raise ValueError('OBB corners must form a nondegenerate convex polygon in perimeter order')
    return xy


def parse(line, prediction=False):
    text = line.strip()
    if not text or text.startswith('#'):
        return None
    cells = tokens(text)
    expected = 10 if prediction else 9
    if len(cells) != expected:
        raise ValueError('Expected class + eight coordinates' + (' + confidence' if prediction else ' (no confidence/difficult field)'))
    result = dict(class_name=class_token(cells[0]), coordinates=coordinates(cells[1:9]))
    if prediction:
        score = float(cells[9])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Confidence must be finite and in [0,1]')
        result['score'] = score
    return result


def format_line(label, xy, score=None):
    label = class_token(label)
    # Quoting preserves multi-word names as one class field.
    label = json.dumps(label, ensure_ascii=False) if re.search(r'\s|["\\]', label) or label.startswith('#') else label
    values = coordinates(xy)
    text = label + ' ' + ' '.join(format(v, '.17g') for v in values)
    if score is not None:
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Confidence must be finite and in [0,1]')
        text += ' ' + format(score, '.17g')
    return text


def load_class_map(root=None, path=None):
    explicit = path is not None
    path = Path(path) if path else Path(root)/'class_map.json' if root else None
    if path is None or not path.exists():
        if explicit:
            raise FileNotFoundError(path)
        return {}
    mapping = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(mapping, dict):
        raise ValueError('class_map.json must map external class tokens to model label names')
    mapping = {class_token(k): class_token(v) for k, v in mapping.items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Class mapping must be one-to-one for unambiguous prediction export')
    return mapping


def export_predictions(coco, rows, destination, class_map=None):
    """One UTF-8 TXT per input image, including empty results; no HBB output."""
    destination = Path(destination)
    inverse = {v: k for k, v in (class_map or {}).items()}
    outputs = {}
    used = set()
    for im in coco['images']:
        name = str(im['file_name']).replace('\\', '/')
        rel = Path(name).with_suffix('.txt')
        if rel.is_absolute() or '..' in rel.parts or ':' in name:
            raise ValueError('Unsafe image path: ' + name)
        if rel.as_posix().casefold() in used or im['id'] in outputs:
            raise ValueError('Duplicate output label path/image ID: ' + name)
        used.add(rel.as_posix().casefold())
        outputs[im['id']] = (rel, [])
    for row in rows:
        if row['image_id'] not in outputs:
            raise ValueError('Prediction references unknown image')
        label = row.get('label', 'vehicle')
        if inverse:
            if label not in inverse:
                raise ValueError('Prediction label has no external class mapping: ' + label)
            label = inverse[label]
        xy = [v for point in row['quad'] for v in point]
        outputs[row['image_id']][1].append(format_line(label, xy, row['score']))
    # Validate every prediction before writing any output.
    for rel, lines in outputs.values():
        p = destination/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8', newline='\n')
    return dict(images=len(outputs), targets=len(rows), format='class x1 y1 x2 y2 x3 y3 x4 y4 confidence',
                confidence='detector/postprocessing score; unchanged by classifier', class_map=class_map or {})
