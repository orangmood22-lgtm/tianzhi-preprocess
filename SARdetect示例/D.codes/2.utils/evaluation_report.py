"""Compact, offline evaluation report with deterministic random high/low examples."""
import collections
import csv
import html
import random
from pathlib import Path
from project_io import write


def evaluate(coco, rows, tax, cf, meta):
    from competition_metrics import evaluate as metric, match, reward, action_label
    if not meta['labels_available']:
        return dict(images=len(coco['images']),predictions=len(rows),det_score=None,class_score=None,joint_score=None,
                    labels_available=False,per_image=[],per_class=[],confusions=[])
    gt = collections.defaultdict(list); pp = collections.defaultdict(list)
    for r in coco['annotations']: gt[r['image_id']].append(r)
    for r in rows: pp[r['image_id']].append(r)
    per = []; classes = {}; confusion = collections.Counter()
    for im in coco['images']:
        gg, preds = gt[im['id']],pp[im['id']]; pairs = match(preds,gg,.3)
        found = {k:j for j,k,_ in pairs}; tp = len(pairs); fp = len(preds)-tp; fn = len(gg)-tp
        per.append(dict(image_id=im['id'],file_name=im['file_name'],gt=len(gg),pred=len(preds),tp=tp,fp=fp,fn=fn,
                        det_score=5*tp/(5*tp+fn+4*fp) if tp+fp+fn else 1.,class_score=None,joint_score=None))
        if meta['classification_labels']:
            for k,g in enumerate(gg):
                depth = g['mixed_label_depth']; node = (g['l1'],g['l2'],g['l3'])
                name = action_label((depth,node),tax)
                c = classes.setdefault(name,dict(label=name,depth=depth,gt=0,detected=0,missed=0,points=0.,wrong_l1=0,wrong_l2=0,wrong_l3=0))
                c['gt'] += 1
                if k not in found:
                    c['missed'] += 1; continue
                p = preds[found[k]]; pd,pn = p['label_depth'],p['label_ids']; c['detected'] += 1
                c['points'] += reward((pd,pn),node,depth)
                err = 'wrong_l1' if pn[0]!=node[0] else 'wrong_l2' if min(pd,depth)>=2 and pn[1]!=node[1] else 'wrong_l3' if pd==depth==3 and pn[2]!=node[2] else None
                if err:
                    c[err] += 1; confusion[(name,p['label'],err)] += 1
    totals = {k:sum(r[k] for r in per) for k in ('tp','fp','fn')}
    result = dict(det_score=sum(r['det_score'] for r in per)/len(per),class_score=None,joint_score=None,
                  **totals,images=len(per),per_image=per,classification_labels=meta['classification_labels'],labels_available=True)
    if meta['classification_labels']:
        result = metric(coco,rows,tax,0.,cf['margin'],'mixed',cf['depth_prior'],cf.get('output_depth'))
        names = {im['id']:im['file_name'] for im in coco['images']}
        for r in result['per_image']: r['file_name'] = names[r['image_id']]
        result['classification_labels'] = True; result['labels_available'] = True
    for c in classes.values():
        c['reward_per_gt'] = c['points']/c['gt']; c['matched_reward'] = c['points']/c['detected'] if c['detected'] else None
    result['class_diagnostics'] = sorted(classes.values(),key=lambda r:(r['reward_per_gt'],-r['gt'],r['label']))
    result['confusions'] = [dict(truth=a,prediction=b,error=e,count=n) for (a,b,e),n in confusion.most_common()]
    result['recall'] = totals['tp']/max(1,totals['tp']+totals['fn'])
    result['precision'] = totals['tp']/max(1,totals['tp']+totals['fp'])
    return result


def csv_write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sample_rows(rows, count, seed):
    """Nonempty GT only, disjoint rank quartiles; ties explicitly randomized by seed."""
    eligible = [r for r in rows if r['gt']>0]
    rng = random.Random(seed); rng.shuffle(eligible)
    score = lambda r: r['joint_score'] if r.get('joint_score') is not None else r['det_score']
    eligible.sort(key=score)
    if len(eligible)<2:
        return dict(low=eligible[:count],high=[])
    n = max(1,len(eligible)//4)
    return {name:rng.sample(pool,min(count,len(pool))) for name,pool in [('low',eligible[:n]),('high',eligible[-n:])]}


def draw(path, im, gt, preds, image_root):
    from PIL import Image, ImageDraw
    image = Image.open(image_root/im['file_name']).convert('RGB')
    # Draw after downscaling so line widths and IDs stay legible on large slices.
    scale = min(1.,1000/max(image.size)); image = image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))))
    canvas = ImageDraw.Draw(image)
    for objects,color,prefix in [(gt,'#40da80','GT'),(preds,'#ff6060','P')]:
        for r in objects:
            q = [(x*scale,y*scale) for x,y in r['quad']]
            canvas.line(q+[q[0]],fill=color,width=2)
            text = prefix
            if prefix=='P':
                text += (f" L{r['label_depth']}:{r['category_id']}" if 'label_depth' in r else '') + f" {r['score']:.2f}"
            elif 'mixed_label_depth' in r:
                d = r['mixed_label_depth']; text += f" L{d}:{r['l'+str(d)]}"
            bounds = canvas.textbbox((0,0),text); tw,th = bounds[2]-bounds[0],bounds[3]-bounds[1]
            tx = max(0,min(min(x for x,y in q),image.width-tw-4))
            ty = min(y for x,y in q)-th-5 if prefix=='GT' else max(y for x,y in q)+2
            ty = max(0,min(ty,image.height-th-4))
            canvas.rectangle((tx,ty,tx+tw+4,ty+th+4),fill='black')
            canvas.text((tx+2,ty+2-bounds[1]),text,fill=color)
    image.save(path,quality=90)


def report(out, results, predictions, coco, image_root, meta, examples, seed, tax):
    esc = html.escape
    summary = [{**dict(model=k),**{f:r.get(f) for f in ('images','joint_score','det_score','class_score','tp','fn','fp','precision','recall')}} for k,r in results.items()]
    write(out/'summary.json',summary); csv_write(out/'summary.csv',summary)
    write(out/'label_ids.json',dict(level1=tax['level1'],level2=tax['level2'],leaves=tax['leaves']))
    ims = {im['id']:im for im in coco['images']}; gt = collections.defaultdict(list)
    for g in coco['annotations']: gt[g['image_id']].append(g)
    cards = []; table = []; selected = {}
    fmt = lambda x: 'N/A' if x is None else f'{100*x:.2f}%'
    for key,r in results.items():
        csv_write(out/key/'per_image.csv',r['per_image'])
        csv_write(out/key/'per_class.csv',r.get('class_diagnostics',[]))
        csv_write(out/key/'confusions.csv',r.get('confusions',[]))
        table.append('<tr><td>'+esc(key)+'</td>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in ('joint_score','det_score','class_score'))+
                     ''.join('<td>'+str(r.get(k,'N/A'))+'</td>' for k in ('fn','fp'))+'</tr>')
        if meta['labels_available']:
            pools = sample_rows(r['per_image'],examples,seed)
        else:
            pools = dict(random=[dict(image_id=i) for i in random.Random(seed).sample(list(ims),min(examples,len(ims)))])
        selected[key] = pools; pp = collections.defaultdict(list)
        for row in predictions[key]: pp[row['image_id']].append(row)
        tiles = []
        if examples:
            dest = out/key/'examples'; dest.mkdir(parents=True,exist_ok=True)
            (dest/'README.txt').write_text(f'Random evaluation examples\nRun: {out.name}; model/group: {key}; split: {meta["split"]}\n'
                'Green: ground truth. Red: final predicted OBB and hierarchical ID. IDs are zero-based within each depth.\n'
                'High/low are random samples from disjoint ranked quartiles of nonempty images; ties randomized.\n'
                'Unlabeled inference uses random images without score claims. See metrics.json, samples.json and label_ids.json.\n',encoding='utf-8')
            for group, rows in pools.items():
                for s in rows:
                    i = s['image_id']; name = f'{group}_{i}.jpg'; draw(dest/name,ims[i],gt[i],pp[i],image_root)
                    caption = f'{group} | '+ims[i]['file_name']
                    if 'gt' in s:
                        caption += f" | score {fmt(s.get('joint_score') if s.get('joint_score') is not None else s.get('det_score'))} | FN {s['fn']} FP {s['fp']}"
                    tiles.append(f'<figure><img loading="lazy" src="{esc(key)}/examples/{name}"><figcaption>{esc(caption)}</figcaption></figure>')
        links = ' · '.join(f'<a href="{esc(key)}/{file}">{label}</a>' for file,label in
                         [('predictions.json','预测'),('metrics.json','指标'),('per_image.csv','逐图明细')])
        if r.get('class_diagnostics'):
            links += f' · <a href="{esc(key)}/per_class.csv">逐类明细</a> · <a href="{esc(key)}/confusions.csv">错误混淆</a>'
        cards.append(f'<details><summary>{esc(key)}：抽样与详情</summary><p>{links}</p><div class="grid">'+''.join(tiles)+'</div></details>')
    write(out/'samples.json',dict(seed=seed,method='random without replacement within disjoint ranked quartiles, nonempty GT; ties shuffled',models=selected))
    limitation = '功能抽样：仅评估前 N 张，不代表全量成绩。' if meta['limited'] else ''
    if not meta['labels_available']: limitation += '无真值：仅输出预测，不计算分数。'
    elif not meta['classification_labels']: limitation += ('仅检查检测与旋转框：分类未执行，分类与总分为 N/A。' if meta.get('detection_only') else '只有车辆检测真值：分类与总分为 N/A；分类预测仍输出。')
    content = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SARdetect 测试结果</title><style>' \
        'body{font:16px system-ui;background:#f5f5f5;color:#222;max-width:1100px;margin:40px auto;padding:0 20px}' \
        'table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}' \
        'td:first-child,th:first-child{text-align:left}details{margin-top:16px;padding:16px;background:white}' \
        'summary{cursor:pointer;font-weight:600}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}' \
        'figure{margin:0}img{width:100%}figcaption{font-size:12px;overflow-wrap:anywhere;padding:8px}a{color:#333}</style><body>'
    split_label = meta['split'] or '无标签推理'
    content += f'<h1>SARdetect 测试结果</h1><p>{meta["images"]} 张图 · {meta["targets"]} 个真值目标 · {esc(split_label)}</p><p>{limitation}</p>'
    content += '<table><thead><tr><th>模型 / 组合</th><th>总分</th><th>检测</th><th>分类</th><th>漏检 FN</th><th>虚警 FP</th></tr></thead><tbody>'+''.join(table)+'</tbody></table>'
    content += '<p>旋转框 IoU ≥ 0.3；逐图平均。绿框为真值，红框为预测；标签编号见 <a href="label_ids.json">分级表</a>。高低分样例来自非空图像的分数两端，各池内随机抽样。</p>'
    content += ''.join(cards)+'<p><a href="request.json">运行配置与权重</a> · <a href="summary.csv">汇总表</a></p></body></html>'
    (out/'report.html').write_text(content,encoding='utf-8')
