"""Proxy competition metric v3.2. Cross-depth wrong L2 under correct L1 earns 0.2."""
import json
from urllib.parse import unquote
from pathlib import Path
import numpy as np
import cv2

DEFAULT_DEPTH_PRIOR=(.1,.3,.6)

def validate_depth_prior(prior):
 values=np.asarray(prior,dtype=np.float64)
 if values.shape!=(3,) or not np.isfinite(values).all() or (values<0).any():
  raise ValueError('depth_prior must contain three finite non-negative probabilities: P1 P2 P3')
 if not np.isclose(values.sum(),1.,rtol=0.,atol=1e-8):
  raise ValueError('depth_prior probabilities must sum to 1; no implicit normalization')
 return tuple(float(x) for x in values)

def _encode_name(name):return name.replace('%','%25').replace('-','%2D')

def parse_label(label):
 """Repeated placeholders encode depth, not additional child classes."""
 if not isinstance(label,str):raise ValueError('Label must be a three-part string')
 raw=tuple(label.split('-'));parts=tuple(unquote(p,errors='strict') for p in raw)
 if len(parts)!=3 or any(not p or p!=p.strip() or _encode_name(p)!=r for p,r in zip(parts,raw)):
  raise ValueError('Use exactly three nonempty components separated by hyphens')
 a,b,c=parts
 if a==b==c:return 1,(a,None,None)
 if a==b:raise ValueError('Invalid level-one placeholder: use A-A-A')
 if b==c:return 2,(a,b,None)
 return 3,parts

def format_label(depth,node):
 if depth not in (1,2,3) or len(node)!=3:raise ValueError('Invalid label depth/path')
 active=node[:depth]
 if any(not isinstance(p,str) or not p or p!=p.strip() for p in active):
  raise ValueError('Active label names must be nonempty and trimmed')
 if depth>=2 and node[0]==node[1] or depth==3 and node[1]==node[2]:
  raise ValueError('Child names must differ from parent names; repetition marks placeholders')
 parts=(node[0],)*3 if depth==1 else (node[0],node[1],node[1]) if depth==2 else tuple(node)
 return '-'.join(_encode_name(p) for p in parts)

def action_label(action,tax):
 depth,node=action;a=tax['level1'][node[0]]
 b=tax['level2'][node[1]]['name'] if depth>=2 else None
 c=next(r['level3'] for r in tax['leaves'] if r['l3']==node[2]) if depth==3 else None
 return format_label(depth,(a,b,c))

def label_action(label,tax):
 # Match full taxonomy paths rather than infer ancestry from prefixes.
 for action in actions(tax):
  if action_label(action,tax)==label:return action
 raise ValueError('Label is absent from this taxonomy: '+str(label))

def score_labels(predicted,truth,tax=None):
 if tax is None:pd,pn=parse_label(predicted);td,tn=parse_label(truth)
 else:(pd,pn),(td,tn)=label_action(predicted,tax),label_action(truth,tax)
 return reward((pd,pn),tn,td)

def reward(pred, truth, depth):
 d,node=pred
 # User rule: wrong level-one ancestor earns zero at every output depth.
 if node[0]!=truth[0]:return 0.
 if d==depth:
  if d==1:return 1.
  if node[1]!=truth[1]:return .2
  if d==2:return 1.
  return 1. if node[2]==truth[2] else .7
 if d>=2 and depth>=2 and node[1]!=truth[1]:return .2
 return {(1,2):.6,(1,3):.4,(2,1):.8,(2,3):.8,(3,1):.6,(3,2):.8}[d,depth]

def actions(tax):
 return [(1,(i,-1,-1)) for i in range(len(tax['level1']))]+[(2,(r['parent'],i,-1)) for i,r in enumerate(tax['level2'])]+[(3,(r['l1'],r['l2'],r['l3'])) for r in tax['leaves']]

def reward_matrix(tax,prior=DEFAULT_DEPTH_PRIOR):
 prior=validate_depth_prior(prior)
 aa=actions(tax)
 return np.asarray([[sum(prior[d-1]*reward(a,(r['l1'],r['l2'],r['l3']),d) for d in (1,2,3)) for r in tax['leaves']] for a in aa],np.float32)

def decode(probs,tax,margin=0.,prior=DEFAULT_DEPTH_PRIOR,output_depth=None):
 if output_depth not in (None,1,2,3):raise ValueError('output_depth must be 1, 2, 3, or None')
 aa=actions(tax); utility=np.asarray(probs)@reward_matrix(tax,prior).T
 allowed=[i for i,a in enumerate(aa) if output_depth is None or a[0]==output_depth]
 # Among actions within margin of maximum expected utility, prefer the coarsest.
 out=[]
 for u in utility:
  best=u.max() if output_depth is None else max(u[allowed])
  candidates=np.flatnonzero(u>=best-margin-1e-8) if output_depth is None else [i for i in allowed if u[i]>=best-margin-1e-8]
  depth=min(aa[i][0] for i in candidates); candidates=[i for i in candidates if aa[i][0]==depth]
  out.append(int(max(candidates,key=lambda i:u[i])))
 return np.asarray(out),utility

def poly_iou(a,b):
 a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
 inter,_=cv2.intersectConvexConvex(a,b);union=cv2.contourArea(a)+cv2.contourArea(b)-inter
 return float(inter/max(union,1e-9))

def match(preds,gt,threshold=.3):
 remaining=set(range(len(gt)));result=[]
 for j in sorted(range(len(preds)),key=lambda i:-preds[i]['score']):
  overlaps=[(poly_iou(preds[j]['quad'],gt[k]['quad']),k) for k in sorted(remaining)]
  if overlaps and max(overlaps)[0]>=threshold:
   iou,k=max(overlaps);remaining.remove(k);result.append((j,k,iou))
 return result

def evaluate(coco,predictions,tax,score=.25,margin=0.,depth_mode='mixed',prior=DEFAULT_DEPTH_PRIOR,output_depth=None):
 aa=actions(tax);gtby={im['id']:[] for im in coco['images']};pb={i:[] for i in gtby}
 for g in coco['annotations']:gtby[g['image_id']].append(g)
 for p in predictions:
  if p['score']>=score:pb[p['image_id']].append(p)
 prior=validate_depth_prior(prior)
 rows=[];perclass={i:dict(gt=0,detected=0,points=0.) for i in range(len(tax['leaves']))}
 for im in coco['images']:
  gt=gtby[im['id']];pred=pb[im['id']];pairs=match(pred,gt);tp=len(pairs);fp=len(pred)-tp;fn=len(gt)-tp
  det=5*tp/(5*tp+fn+4*fp) if tp+fp+fn else 1.
  total=0.;leafcorrect=0;points=[]
  for g in gt:
   if g['l3']>=0:perclass[g['l3']]['gt']+=1
  for j,k,iou in pairs:
   g=gt[k];p=pred[j]; depth=g['mixed_label_depth'] if depth_mode=='mixed' else int(depth_mode)
   action=aa[int(decode(np.asarray([p['probabilities']]),tax,margin,prior,output_depth)[0][0])] if 'probabilities' in p else (1,(-1,-1,-1))
   s=reward(action,(g['l1'],g['l2'],g['l3']),depth);total+=s;points.append(s)
   if g['l3']>=0:
    perclass[g['l3']]['detected']+=1;perclass[g['l3']]['points']+=s
   if 'probabilities' in p:leafcorrect+=int(np.argmax(p['probabilities'])==g['l3'])
  cls=total/len(gt) if gt else (1. if not pred else 0.)
  rows.append(dict(image_id=im['id'],source_id=im.get('source_id'),gt=len(gt),pred=len(pred),tp=tp,fp=fp,fn=fn,det_score=det,class_score=cls,matched_class_score=total/tp if tp else 0.,joint_score=.5*(det+cls),matched_leaf_correct=leafcorrect))
 keys=['det_score','class_score','matched_class_score','joint_score']
 result=dict(**{k:float(np.mean([r[k] for r in rows])) for k in keys},images=len(rows),tp=sum(r['tp'] for r in rows),fp=sum(r['fp'] for r in rows),fn=sum(r['fn'] for r in rows),threshold=score,margin=margin,depth_mode=depth_mode,depth_prior=list(prior),label_format='repeat-parent-v1',per_image=rows,per_class=perclass,unknown_fine_targets=sum(g['l3']<0 for g in coco['annotations']),per_class_scope='fine-label targets only; all declared depths included in aggregate scores',protocol='proxy-v3.2: class sum / all GT, missed=0, empty correct=1, wrong L1=0, cross-depth wrong L2 under correct L1=0.2, FP penalized by detection')
 if output_depth is not None:result['output_depth']=output_depth
 return result

def tests():
 root=(0,-1,-1);parent=(0,1,-1);leaf=(0,1,2);other=(0,1,3);wrong=(0,2,4)
 expected=[[1,.6,.4],[.8,1,.8],[.6,.8,1]]
 for i,node in enumerate([root,parent,leaf]):
  for d in range(1,4):assert reward((i+1,node),leaf,d)==expected[i][d-1]
 assert reward((3,other),leaf,3)==.7 and reward((3,wrong),leaf,3)==.2
 assert reward((2,wrong),leaf,2)==.2 and reward((3,(1,2,4)),leaf,3)==0
 assert reward((2,(0,2,-1)),leaf,3)==.2 and reward((3,wrong),leaf,2)==.2
 assert reward((2,(1,2,-1)),leaf,3)==0 and reward((3,(1,2,4)),leaf,2)==0
 q=cv2.boxPoints(((20,20),(20,8),37));assert abs(poly_iou(q,q)-1)<1e-5;assert poly_iou(q,q+100)==0
 assert len(match([dict(quad=q,score=.9),dict(quad=q,score=.8)],[dict(quad=q)]))==1
 tax=dict(level1=['vehicle'],level2=[dict(parent=0)],leaves=[dict(l1=0,l2=0,l3=0)])
 coco=dict(images=[dict(id=1),dict(id=2)],annotations=[dict(image_id=1,quad=q,l1=0,l2=0,l3=0,mixed_label_depth=3)])
 pp=[dict(image_id=1,quad=q,score=.9,probabilities=[1.]),dict(image_id=1,quad=q,score=.8,probabilities=[1.])]
 result=evaluate(coco,pp,tax,depth_mode='3',prior=(0.,0.,1.))
 # Duplicate is an FP; macro mean includes the correctly empty second image.
 assert abs(result['det_score']-(5/9+1)/2)<1e-7 and result['class_score']==1.
 missed=evaluate(coco,[],tax,depth_mode='3',prior=(0.,0.,1.))
 assert missed['det_score']==.5 and missed['class_score']==.5
 pp.append(dict(image_id=2,quad=q,score=.7,probabilities=[1.]))
 false_empty=evaluate(coco,pp,tax,depth_mode='3',prior=(0.,0.,1.))
 assert abs(false_empty['det_score']-5/18)<1e-7 and false_empty['class_score']==.5
 print('reward table, rotated IoU, one-to-one matching, per-image averaging and empty-image conventions passed')
if __name__=='__main__':tests()
