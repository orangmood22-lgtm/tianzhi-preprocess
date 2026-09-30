"""Deterministic, group-aware multilabel split and UTF-8 dataset statistics."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,random,re,sys,tempfile,time
from pathlib import Path
from urllib.parse import unquote
IMAGE_EXTS={'.tif','.tiff','.img','.png','.jpg','.jpeg','.bmp','.webp'}
SPLITS=('train','val','test')

def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def label_parts(label):
 raw=label.split('-');parts=[unquote(x,errors='strict') for x in raw]
 if len(parts)!=3 or any(not x or x.strip()!=x or x.replace('%','%25').replace('-','%2D')!=r for x,r in zip(parts,raw)):raise ValueError('非法三级命名：'+label)
 a,b,c=parts
 if a==b==c:return 1,parts
 if a==b:raise ValueError('一级占位必须为 A-A-A：'+label)
 return (2 if b==c else 3),parts

def parse_line(line,context,class_map=None,label_format="label-first"):
 from obb_labels import parse
 if label_format not in ("label-first","auto"):raise ValueError("Only class-first competition labels are accepted")
 try:
  obj=parse(line)
  if obj is None:return None
  external=obj['class_name'];label=(class_map or {}).get(external,external)
  try:depth,parts=label_parts(label)
  except ValueError:depth,parts=0,[label,label,label]
  return dict(label=label,class_name=external,depth=depth,parts=parts,coordinates=obj['coordinates'],difficult=0)
 except (ValueError,TypeError) as exc:raise ValueError(context+': '+str(exc)) from exc


def capacities(n,ratios):
 values=[n*r for r in ratios];out=[int(v) for v in values]
 for i in sorted(range(3),key=lambda i:(-(values[i]-out[i]),i))[:n-sum(out)]:out[i]+=1
 active=[i for i,r in enumerate(ratios) if r>0]
 if n>=len(active):
  for i in active:
   if out[i]==0:
    j=max(active,key=lambda j:out[j]);out[j]-=1;out[i]+=1
 return out

def load_dataset(root,args):
 images=sorted(p for p in (root/'images').rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
 if not images:raise ValueError('images中没有支持的图像')
 labels={p.relative_to(root/'labels').with_suffix('').as_posix():p for p in (root/'labels').rglob('*.txt') if p.is_file()}
 if len(labels)!=len(list((root/'labels').rglob('*.txt'))):raise ValueError('重复标签路径')
 from obb_labels import load_class_map
 class_map=load_class_map(root,getattr(args,'class_map',None))
 by_key={};rows=[];parents={};parent=list(range(len(images)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):
  a,b=find(i),find(j)
  if a!=b:parent[max(a,b)]=min(a,b)
 hashes={};sources={};pattern=re.compile(args.group_regex) if args.group_regex else None;aggregate=hashlib.sha256()
 for i,path in enumerate(images):
  key=path.relative_to(root/'images').with_suffix('').as_posix()
  if '\n' in key or '\r' in key:raise ValueError('图像文件名不得包含换行符')
  if key in by_key:raise ValueError('同一相对stem对应多个图像扩展名：'+key)
  by_key[key]=i;annotation=labels.get(key)
  if not annotation:raise ValueError('缺少标签文件：'+key+'；负样本请提供空TXT')
  objects=[]
  for n,line in enumerate(annotation.read_text(encoding='utf-8-sig').splitlines(),1):
   obj=parse_line(line,f'{annotation}:{n}',class_map,args.label_format)
   if obj:objects.append(obj)
  digest=sha_file(path);annotation_hash=sha_file(annotation);aggregate.update((key+'\0'+digest+'\0'+annotation_hash+'\n').encode())
  signature=sorted((o['label'],tuple(o['coordinates']),o['difficult']) for o in objects)
  if digest in hashes:
   j=hashes[digest]
   if signature!=rows[j]['signature']:raise ValueError('内容相同的图像标注不一致：'+key)
   union(i,j)
  else:hashes[digest]=i
  source=None
  if pattern:
   match=pattern.search(path.relative_to(root/'images').as_posix())
   if not match:raise ValueError('--group-regex未匹配：'+key)
   source=match.groupdict().get('source') if 'source' in match.groupdict() else match.group(1) if match.lastindex else match.group(0)
   if not source:raise ValueError('--group-regex必须提取非空来源标识：'+key)
   if source in sources:union(i,sources[source])
   else:sources[source]=i
  features=collections.Counter({'images':1,'objects':len(objects),'empty':int(not objects)})
  exact=collections.Counter(o['label'] for o in objects)
  for label,count in exact.items():features['class:'+label]=count;features['presence:'+label]=1
  for obj in objects:
   features['depth:'+str(obj['depth'])]+=1;features['L1:'+obj['parts'][0]]+=1
   if obj['depth']>=2:features['L2:'+obj['parts'][0]+' / '+obj['parts'][1]]+=1
  rows.append(dict(path=path.relative_to(root).as_posix(),objects=objects,features=features,signature=signature,sha256=digest))
  if (i+1)%100==0:print(f'扫描图像与标注：{i+1}/{len(images)}',file=sys.stderr,flush=True)
 if set(labels)-set(by_key):raise ValueError('存在孤立标签：'+', '.join(sorted(set(labels)-set(by_key))[:8]))
 grouped=collections.defaultdict(list)
 for i in range(len(rows)):grouped[find(i)].append(i)
 groups=[]
 for ids in grouped.values():
  f=collections.Counter()
  for i in ids:f.update(rows[i]['features'])
  groups.append(dict(ids=ids,f=f,classes={k for k in f if k.startswith('presence:')}))
 return rows,groups,aggregate.hexdigest(),len(images)-len(hashes)

def stratify(rows,groups,ratios,seed,restarts=6,swaps=2500):
 total=collections.Counter();freq=collections.Counter()
 for g in groups:total.update(g['f']);freq.update(sorted(g['classes']))
 active=[i for i,r in enumerate(ratios) if r>0];cap=capacities(len(rows),ratios);target=[{k:v*r for k,v in total.items()} for r in ratios]
 weights={k:(3 if k.startswith('presence:') else 2 if k.startswith('depth:') else .2 if k=='objects' else 1)/max(v,1) for k,v in total.items()}
 def cost(counts):return sum(weights[k]*(counts[i][k]-target[i][k])**2 for i in active for k in total)+sum(1000 for k,n in freq.items() if n>=len(active) for i in active if counts[i][k]==0)
 best=None
 for attempt in range(restarts):
  rng=random.Random(seed+attempt*104729);order=list(range(len(groups)));rng.shuffle(order);order.sort(key=lambda j:(min([freq[k] for k in groups[j]['classes']] or [10**12]),-len(groups[j]['ids'])))
  counts=[collections.Counter() for _ in range(3)];assigned=[-1]*len(groups);remaining=freq.copy()
  for j in order:
   g=groups[j];f=g['f'];n=len(g['ids']);allowed=[i for i in active if counts[i]['images']+n<=cap[i]]
   if not allowed:
    overflow=min(max(0,counts[i]['images']+n-cap[i]) for i in active);allowed=[i for i in active if max(0,counts[i]['images']+n-cap[i])==overflow]
   # Preserve the only available group of a class in training whenever feasible.
   if 0 in allowed and any(freq[k]==1 for k in g['classes']):allowed=[0]
   must=[]
   for k in g['classes']:
    missing={i for i in active if counts[i][k]==0}
    if freq[k]>=len(active) and missing and remaining[k]<=len(missing):must.append(missing)
   if must:
    feasible=set(allowed).intersection(*must)
    if feasible:allowed=sorted(feasible)
   rng.shuffle(allowed)
   def delta(i):return sum(weights[k]*(2*(counts[i][k]-target[i][k])*v+v*v) for k,v in f.items())
   selected=min(allowed,key=delta);assigned[j]=selected;counts[selected].update(f)
   for k in g['classes']:remaining[k]-=1
  # Equal-size group swaps keep achieved image counts and improve joint balance.
  for _ in range(swaps):
   if len(groups)<2:break
   a,b=rng.sample(range(len(groups)),2);i,j=assigned[a],assigned[b];ga,gb=groups[a],groups[b]
   if i==j or len(ga['ids'])!=len(gb['ids']):continue
   if i==0 and any(freq[k]==1 for k in ga['classes']) or j==0 and any(freq[k]==1 for k in gb['classes']):continue
   keys=sorted(set(ga['f'])|set(gb['f']));delta={k:gb['f'][k]-ga['f'][k] for k in keys}
   if any(freq.get(k,0)>=len(active) and ((counts[i][k]>0 and counts[i][k]+v==0) or (counts[j][k]>0 and counts[j][k]-v==0)) for k,v in delta.items()):continue
   gain=sum(weights[k]*(2*(counts[i][k]-target[i][k])*v+v*v-2*(counts[j][k]-target[j][k])*v+v*v) for k,v in delta.items())
   if gain < -1e-10:
    for k,v in delta.items():counts[i][k]+=v;counts[j][k]-=v
    assigned[a],assigned[b]=j,i
  score=cost(counts)+sum(100*(counts[i]['images']-cap[i])**2 for i in active)
  if best is None or score<best[0]:best=(score,assigned,counts)
 score,assigned,counts=best;parts=[[] for _ in range(3)]
 for g,i in zip(groups,assigned):parts[i].extend(g['ids'])
 for p in parts:p.sort(key=lambda i:rows[i]['path'])
 return parts,dict(objective=score,requested_images=cap,group_class_presence=dict(freq),counts=counts)

def report_text(root,rows,groups,parts,ratios,seed,digest,duplicates,meta,args):
 depth_by_label={o["label"]:o["depth"] for row in rows for o in row["objects"]}
 def describe(ids):
  objects=[o for i in ids for o in rows[i]['objects']];depth=collections.Counter(o['depth'] for o in objects);exact=collections.Counter(o['label'] for o in objects);l1=collections.Counter(o['parts'][0] for o in objects);l2=collections.Counter(o['parts'][0]+' / '+o['parts'][1] for o in objects if o['depth']>=2)
  return dict(images=len(ids),objects=len(objects),empty=sum(not rows[i]['objects'] for i in ids),difficult=sum(o['difficult']>0 for o in objects),depth=depth,exact=exact,l1=l1,l2=l2)
 stats=[describe(range(len(rows)))]+[describe(ids) for ids in parts];lines=['SARdetect 数据集划分与统计',f'数据集：{root.name}',f'随机种子：{seed}',f'目标比例 train/val/test：{ratios}',f'数据内容指纹 SHA256：{digest}',f'切片总数：{len(rows)}',f'目标总数：{stats[0]["objects"]}',f'不同完整标签数：{len(stats[0]["exact"])}',f'空标签切片数：{stats[0]["empty"]}',f'difficult>0目标数：{stats[0]["difficult"]}（统计与分层均包含）',f'相同文件内容重复图像数：{duplicates}（同组，不跨集合）',f'独立分组数：{len(groups)}',f'同源分组正则：{args.group_regex or "未提供；仅自动合并内容完全相同的图像"}',f'算法：稀有类别优先、多标签目标数/出现切片数/层级分层；{args.restarts}次确定性搜索和等大小组交换',f'平衡目标函数值：{meta["objective"]:.8f}','', '集合	切片数	切片占全体比例	目标数	目标占全体比例	空切片数']
 for name,st in zip(['all',*SPLITS],stats):lines.append(f'{name}	{st["images"]}	{st["images"]/len(rows):.6%}	{st["objects"]}	{st["objects"]/max(stats[0]["objects"],1):.6%}	{st["empty"]}')
 lines+=['', '按声明深度统计的不同标签种数：'+str({d:sum(depth_by_label[k]==d for k in stats[0]['exact']) for d in (1,2,3)})]
 lines+=['','标签深度统计（比例分母为对应集合目标总数）','集合	一级数量	一级比例	二级数量	二级比例	三级数量	三级比例']
 for name,st in zip(['all',*SPLITS],stats):lines.append(name+'	'+'	'.join(f'{st["depth"][d]}	{st["depth"][d]/max(st["objects"],1):.6%}' for d in [1,2,3]))
 lines+=['','完整声明标签统计；各集合比例分母为本集合目标总数','标签	深度	全体数量	全体占比	train数量	train占比	val数量	val占比	test数量	test占比	跨集合分布L1偏差']
 deviations=[]
 for label in sorted(stats[0]['exact']):
  n=stats[0]['exact'][label];dev=sum(abs(stats[i+1]['exact'][label]/n-ratios[i]) for i in range(3));deviations.append((dev,label))
  lines.append(label+'	'+str(depth_by_label[label])+'	'+'	'.join(f'{st["exact"][label]}	{st["exact"][label]/max(st["objects"],1):.6%}' for st in stats)+f'	{dev:.6f}')
 for key,title in [('l1','一级祖先类别统计（所有目标均可统计）'),('l2','二级祖先类别统计（仅真实标签深度>=2，不猜测一级标签的细类别）')]:
  lines+=['',title,'类别	all	train	val	test']
  for label in sorted(stats[0][key]):lines.append(label+'	'+'	'.join(str(st[key][label]) for st in stats))
 lines+=['','偏差与限制说明',f'实际切片数：{[len(p) for p in parts]}；按比例取整目标：{meta["requested_images"]}。整组分配可能导致数量偏差。','标签深度比例只是数据统计，不自动改写分类器depth_prior。','同一切片及同组图像只出现于一个集合；不复制/移动原图和标签。','单独存在于一个独立组的类别优先保留在训练集，容量与其他分组约束可能冲突。','稀有类、同图类别共现和同源分组可能使严格均匀划分不可行。','未提供同源规则时无法保证不同切片间不存在空间/采集来源关联；可用--group-regex指定来源分组。']
 active=[i for i,r in enumerate(ratios) if r>0]
 for label in sorted(stats[0]['exact']):
  groups_n=meta['group_class_presence']['presence:'+label];missing=[SPLITS[i] for i in active if stats[i+1]['exact'][label]==0]
  if missing:lines.append(f'缺失类别：{label}；缺少集合={missing}；独立组数={groups_n}；'+('不足以覆盖全部非空比例集合' if groups_n<len(active) else '受容量或共现/分组约束影响'))
 lines+=['','类别分配偏差最高的10类（sum |该类在集合中占比-目标比例|）：']+[f'{label}	{dev:.6f}' for dev,label in sorted(deviations,reverse=True)[:10]]
 return '\n'.join(lines)+'\n'

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dataset',required=True,help='A.dataset下的单个数据集文件夹名');ap.add_argument('--ratios',nargs=3,type=float,required=True,metavar=('TRAIN','VAL','TEST'));ap.add_argument('--seed',type=int,required=True);ap.add_argument('--group-regex',help='按相对图像文件名提取同源标识，使用命名组source或首个捕获组');ap.add_argument('--class-map',type=Path);ap.add_argument('--label-format',choices=['label-first'],default='label-first');ap.add_argument('--restarts',type=int,default=6);ap.add_argument('--swaps',type=int,default=2500);ap.add_argument('--force',action='store_true',help='明确允许覆盖既有train/val/test/data.txt，备份放入C.results');ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2],help=argparse.SUPPRESS);args=ap.parse_args(argv)
 try:
  if Path(args.dataset).name!=args.dataset or args.dataset in ('.','..') or '/' in args.dataset or '\\' in args.dataset:raise ValueError('--dataset必须为单个文件夹名')
  if not all(math.isfinite(x) and x>=0 for x in args.ratios) or not math.isclose(sum(args.ratios),1,rel_tol=0,abs_tol=1e-8):raise ValueError('三个划分比例必须非负且和为1')
  if args.ratios[0]<=0:raise ValueError('训练比例必须大于0')
  if args.restarts<1 or args.swaps<0:raise ValueError('restarts必须>=1，swaps必须>=0')
  project=args.project_root.resolve();root=(project/'A.dataset'/args.dataset).resolve(strict=True)
  if not root.is_relative_to((project/'A.dataset').resolve()):raise ValueError('数据集符号链接指向A.dataset之外；请在参赛目录中使用独立数据集')
  if not (root/'images').is_dir() or not (root/'labels').is_dir():raise ValueError('数据集须包含images和labels目录')
  names=['train.txt','val.txt','test.txt','data.txt'];existing=[root/n for n in names if (root/n).exists()]
  if any((root/n).exists() for n in ['train','val','test']):raise ValueError('发现无扩展名旧划分文件，请先明确统一为train.txt/val.txt/test.txt，避免两套划分并存')
  if existing and not args.force:raise ValueError('划分或统计文件已存在；使用--force明确覆盖，旧文件将备份')
  rows,groups,digest,duplicates=load_dataset(root,args);parts,meta=stratify(rows,groups,args.ratios,args.seed,args.restarts,args.swaps)
  assert sorted(i for ids in parts for i in ids)==list(range(len(rows)))
  sets=[{rows[i]['sha256'] for i in ids} for ids in parts];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
  texts=[''.join(rows[i]['path']+'\n' for i in ids) for ids in parts];texts.append(report_text(root,rows,groups,parts,args.ratios,args.seed,digest,duplicates,meta,args))
  run=project/'C.results/0.settings/dataset_split'/args.dataset/(str(time.time_ns())+'-s'+str(args.seed));run.mkdir(parents=True)
  for p in existing:(run/p.name).write_bytes(p.read_bytes())
  (run/'request.json').write_text(json.dumps(dict(dataset=args.dataset,ratios=args.ratios,seed=args.seed,group_regex=args.group_regex,label_format=args.label_format,data_sha256=digest,old_files=[p.name for p in existing],class_map_sha256=sha_file(args.class_map) if args.class_map else None),ensure_ascii=False,indent=2),encoding='utf-8')
  temporaries=[]
  try:
   for name,text in zip(names,texts):
    fd,path=tempfile.mkstemp(prefix='.'+name+'.',dir=root);temporaries.append(Path(path))
    with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as f:f.write(text)
   for path,name in zip(temporaries,names):path.replace(root/name)
  except BaseException:
   for name in names:
    if (run/name).exists():(root/name).write_bytes((run/name).read_bytes())
    elif (root/name).exists():(root/name).unlink()
   raise
  finally:
   for p in temporaries:
    if p.exists():p.unlink()
  summary=dict(status='complete',images=len(rows),objects=sum(len(r['objects']) for r in rows),splits=dict(zip(SPLITS,map(len,parts))),data_txt=str(root/'data.txt'),output_hashes={n:sha_file(root/n) for n in names},audit=str(run));(run/'completed.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2));return summary
 except (ValueError,OSError,re.error) as e:ap.error(str(e))
if __name__=='__main__':main()
