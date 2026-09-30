"""Read split lists and convert flat HBB or competition OBB labels to detector COCO."""
from pathlib import Path
import json,math
def read_split(dataset_root,split):
 root=Path(dataset_root);rows=[x.strip() for x in (root/(split+'.txt')).read_text(encoding='utf-8').splitlines() if x.strip()];images=[(root/x).resolve(strict=True) for x in rows]
 if len(images)!=len(set(images)):raise ValueError('Duplicate image in split')
 if any(not p.is_relative_to((root/'images').resolve()) for p in images):raise ValueError('Image outside dataset images/')
 return images
def split_overlap_audit(dataset_root):
 sets={s:set(read_split(dataset_root,s)) for s in ['train','val','test']}
 return {f'{a}_{b}':len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
def write_coco(dataset_root,split,destination):
 from PIL import Image
 from competition_split import parse_line
 from obb_labels import load_class_map
 root=Path(dataset_root);mapping=load_class_map(root);c={'images':[],'annotations':[],'categories':[{'id':1,'name':'vehicle','supercategory':'vehicle'}]}
 for ident,p in enumerate(read_split(root,split),1):
  name=p.relative_to((root/'images').resolve());lab=root/'labels'/name.with_suffix('.txt')
  with Image.open(p) as im:w,h=im.size
  c['images'].append(dict(id=ident,file_name=name.as_posix(),width=w,height=h))
  for n,line in enumerate(lab.read_text(encoding='utf-8-sig').splitlines(),1):
   line=line.strip()
   if not line or line.startswith('#') or line.lower().startswith(('imagesource:','gsd:')):continue
   obj=parse_line(line,f'{lab}:{n}',mapping);quad=[obj['coordinates'][i:i+2] for i in range(0,8,2)];xx=obj['coordinates'][::2];yy=obj['coordinates'][1::2];x0,y0=max(0,min(xx)),max(0,min(yy));x1,y1=min(w,max(xx)),min(h,max(yy))
   if x1<=x0 or y1<=y0:raise ValueError(f'Empty/outside box {lab}:{n}')
   c['annotations'].append(dict(id=len(c['annotations'])+1,image_id=ident,category_id=1,bbox=[x0,y0,x1-x0,y1-y0],area=(x1-x0)*(y1-y0),iscrowd=0))
   if quad is not None:
    a=c['annotations'][-1];a.update(quad=quad,segmentation=[[v for xy in quad for v in xy]],area=abs(sum(quad[i][0]*quad[(i+1)%4][1]-quad[(i+1)%4][0]*quad[i][1] for i in range(4)))/2)
 Path(destination).parent.mkdir(parents=True,exist_ok=True);Path(destination).write_text(json.dumps(c,ensure_ascii=False)+'\n',encoding='utf-8');return {'images':len(c['images']),'instances':len(c['annotations']),'path':str(destination)}
