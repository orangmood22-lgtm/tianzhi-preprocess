"""Convert the competition's flat OBB TXT layout outside A.dataset."""
import hashlib,json
from pathlib import Path
from PIL import Image
from competition_split import parse_line
ROOT=Path(__file__).resolve().parents[2]
def prepare(source,taxonomy):
 from competition_metrics import label_action
 from obb_labels import load_class_map
 source=Path(source).resolve();class_map=load_class_map(source);tax=json.loads(Path(taxonomy).read_text());rows={};h=hashlib.sha256(('class-first-v2:'+str(source)+json.dumps(class_map,sort_keys=True)).encode());seen=set()
 for split in ['train','val','test']:
  p=source/(split+'.txt');h.update(p.read_bytes());rows[split]=[]
  for line in p.read_text().splitlines():
   if not line.strip():continue
   im=(source/line.strip()).resolve(strict=True)
   if not im.is_relative_to((source/'images').resolve()):raise ValueError('Split image outside images/: '+str(im))
   if im in seen:raise ValueError('Duplicate or overlapping split image: '+str(im))
   seen.add(im);name=im.relative_to((source/'images').resolve()).as_posix();lab=source/'labels'/Path(name).with_suffix('.txt');text=lab.read_text(encoding='utf-8-sig');h.update(name.encode()+lab.read_bytes()+str(im.stat().st_mtime_ns).encode()+str(im.stat().st_size).encode());objs=[]
   for n,line in enumerate(text.splitlines(),1):
    obj=parse_line(line,f'{lab}:{n}',class_map)
    if obj:
     depth,ids=label_action(obj['label'],tax);objs.append((obj,depth,ids))
   with Image.open(im) as image:w,hh=image.size
   rows[split].append((name,w,hh,objs))
 h.update(json.dumps(tax,sort_keys=True).encode());digest=h.hexdigest();out=ROOT/'.cache/dataset_cache'/digest
 out.parent.mkdir(parents=True,exist_ok=True)
 import fcntl
 with (out.parent/(digest+'.lock')).open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (out/'VERIFIED.json').exists():return out
  out.mkdir(parents=True,exist_ok=True);(out/'annotations').mkdir(exist_ok=True)
  if not (out/'images').exists():(out/'images').symlink_to((source/'images').resolve(),target_is_directory=True)
  for split,ims in rows.items():
   c={'images':[],'annotations':[],'categories':[{'id':1,'name':'vehicle'}]}
   for ident,(name,w,hh,oo) in enumerate(ims,1):
    c['images'].append(dict(id=ident,file_name=name,width=w,height=hh))
    for obj,depth,ids in oo:
     xy=obj['coordinates'];q=list(map(list,zip(xy[::2],xy[1::2])));xx=xy[::2];yy=xy[1::2];x,y=max(0,min(xx)),max(0,min(yy));bw,bh=min(w,max(xx))-x,min(hh,max(yy))-y
     if min(bw,bh)<=0:raise ValueError('Box outside image: '+name)
     c['annotations'].append(dict(id=len(c['annotations'])+1,image_id=ident,category_id=1,bbox=[x,y,bw,bh],area=bw*bh,iscrowd=0,quad=q,l1=ids[0],l2=ids[1] if depth>=2 else -1,l3=ids[2] if depth==3 else -1,mixed_label_depth=depth,difficult=obj['difficult']))
   (out/f'annotations/{split}.json').write_text(json.dumps(c))
  (out/'taxonomy.json').write_text(json.dumps(tax));(out/'VERIFIED.json').write_text(json.dumps(dict(manifest_sha256=digest,source=str(source),format='competition-flat-obb',unknown_fine_labels='-1; masked in training')))
  return out

def yolo_dataset(raw,dest):
 """Build a bounded, exact COCO-to-YOLO view; never writes source labels/caches."""
 dest=Path(dest)/'yolo_data';dest.mkdir(parents=True,exist_ok=True);source=Path(raw['data']['image_prefix'])
 for split in ['train','val','test']:
  coco=json.loads((Path(raw['data']['root'])/raw['data'][f'{split}_annotations']).read_text());group={}
  for a in coco['annotations']:group.setdefault(a['image_id'],[]).append(a)
  names=[]
  for im in coco['images']:
   file=Path(im['file_name']);p=dest/'images'/split/file;p.parent.mkdir(parents=True,exist_ok=True)
   if not p.exists():p.symlink_to((source/file).resolve(strict=True))
   lab=dest/'labels'/split/file.with_suffix('.txt');lab.parent.mkdir(parents=True,exist_ok=True);lines=[]
   for a in group.get(im['id'],[]):
    x,y,w,h=a['bbox'];iw,ih=im['width'],im['height'];lines.append(f'0 {(x+w/2)/iw:.9f} {(y+h/2)/ih:.9f} {w/iw:.9f} {h/ih:.9f}')
   lab.write_text('\n'.join(lines)+'\n');names.append(str(p))
  (dest/(split+'.txt')).write_text('\n'.join(names)+'\n')
 p=dest/'data.yaml';p.write_text(f'path: {dest}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: vehicle\n');return str(p)
