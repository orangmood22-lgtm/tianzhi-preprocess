"""Project paths, JSON, checksums and deployed asset manifest; no ML imports."""
import hashlib,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
UTIL=ROOT/'D.codes/2.utils'
from training_resources import interpreter
PYTHON=interpreter()
YOLO_PYTHON=interpreter(True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def manifest():return read(Path(os.environ.get('SARDETECT_MANIFEST',str(ROOT/'B.models/v3_manifest.json'))))
def checked(row):
    path=ROOT/row['checkpoint']
    if sha(path)!=row['sha256']:raise ValueError(f'Checkpoint hash mismatch: {path}')
    return path
