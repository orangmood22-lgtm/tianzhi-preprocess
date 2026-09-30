"""One public experiment contract; backend state lives outside C.results runs."""
import collections
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
KINDS = ('1.train', '2.test', '3.infer')
PUBLIC_ITEMS = {'config', 'record.txt', 'result.txt', 'visualization'}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    os.replace(tmp,p)


def work_dir(public):
    public=Path(public).resolve()
    rel=public.relative_to(ROOT/'C.results')
    if len(rel.parts)!=2 or rel.parts[0] not in KINDS:
        raise ValueError('Experiment output must be C.results/{1.train,2.test,3.infer}/RUN')
    return ROOT/'.runtime/experiments'/rel


def public_dir(work):
    work=Path(work).resolve()
    try:rel=work.relative_to(ROOT/'.runtime/experiments')
    except ValueError:return work  # Historical readers may pass a legacy directory.
    if len(rel.parts)!=2 or rel.parts[0] not in KINDS:raise ValueError('Invalid experiment state path')
    return ROOT/'C.results'/rel


def resolve_run(path):
    """Read a public run or an old run without breaking saved resume commands."""
    p=Path(path).resolve();p=p.parent if p.is_file() else p
    if p.is_relative_to(ROOT/'C.results'):
        hidden=work_dir(p)
        return hidden if hidden.exists() else p
    return p


def initialize(public, request, status='running'):
    public=Path(public);public.mkdir(parents=True,exist_ok=True)
    extras={p.name for p in public.iterdir()}-PUBLIC_ITEMS
    if extras:raise ValueError('Migrate legacy output before writing public report: '+str(sorted(extras)))
    (public/'config').mkdir(exist_ok=True)
    write(public/'config/run.json',request)
    (public/'visualization').mkdir(exist_ok=True)
    if not (public/'record.txt').exists():
        (public/'record.txt').write_text('SARdetect 实验记录\n实验：'+public.name+'\n',encoding='utf-8')
    (public/'result.txt').write_text('状态：'+status+'\n运行未完成时不报告最终成绩。\n',encoding='utf-8')


def label(k):
    return {'coco/bbox_mAP':'检测mAP50–95','coco/bbox_mAP_50':'检测AP50','mAP':'检测mAP50–95',
            'AP50':'检测AP50','mixed_score':'分类混合深度得分','full_depth_score':'分类全深度得分',
            'leaf_top1':'细类Top1','val_mean_polygon_iou':'OBB验证平均IoU','mean_iou':'平均IoU',
            'det_score':'检测分','class_score':'分类分','joint_score':'综合分',
            'loss':'训练loss','epoch':'轮次'}.get(k,k)


def numbers(values):
    return '  '.join(f'{label(k)}={v:.6f}' for k,v in values.items()
                     if isinstance(v,(float,int)) and not isinstance(v,bool))


def friendly(line):
    line=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',line)
    if not line.startswith('[SAR] '):return line
    try:r=json.loads(line[6:])
    except ValueError:return line
    who=r.get('model','')+('/'+r['branch'] if r.get('branch') else '')
    head=f'[{r.get("time","")}] {who} '
    kind=r.get('kind')
    if kind=='validation':return head+f'第{r.get("epoch")}轮验证：'+numbers(r.get('metrics',{}))+'\n'
    if kind=='progress':return head+f'训练 {r.get("epoch")}/{r.get("epochs")}轮 '+numbers({k:r[k] for k in ('step','steps','loss','eta_seconds') if k in r})+'\n'
    if kind=='checkpoint':return head+f'第{r.get("epoch")}轮检查点已保存\n'
    if kind=='configuration':return head+'训练配置：'+numbers(r.get('training',r))+'（完整配置见config）\n'
    return head+str(kind)+' '+numbers(r)+'\n'


def append_line(public,line):
    with (Path(public)/'record.txt').open('a',encoding='utf-8') as f:f.write(friendly(line))


def events(work):
    p=Path(work)/'events.jsonl'
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []


def metric_values(m):
    if 'coco_eval_bbox' in m:return {'mAP':m['coco_eval_bbox'][0],'AP50':m['coco_eval_bbox'][1]}
    if 'coco/bbox_mAP' in m:return {'mAP':m['coco/bbox_mAP'],'AP50':m.get('coco/bbox_mAP_50')}
    if 'metrics/mAP50-95(B)' in m:return {'mAP':m['metrics/mAP50-95(B)'],'AP50':m.get('metrics/mAP50(B)')}
    return {k:v for k,v in m.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}


def collect_training(work):
    """Normalize all detector backends and component histories without ML imports."""
    work=Path(work);ee=events(work);groups={}
    for key in sorted({e.get('model') for e in ee if e.get('model')} | {p.name for p in work.glob('model-*') if p.is_dir()}):
        base=work/key
        vals=[e for e in ee if e.get('model')==key and e.get('kind')=='validation' and not e.get('branch')]
        records={e['epoch']:dict(epoch=e['epoch'],**metric_values(e['metrics'])) for e in vals}
        notes=[]
        # Framework logs are sampled smoothed losses. Latest attempt for each epoch wins;
        # the uninterrupted raw console remains available in record.txt.
        for p in sorted(base.rglob('scalars.json')):
            epoch_losses=collections.defaultdict(list)
            for line in p.read_text(encoding='utf-8').splitlines():
                r=json.loads(line)
                if 'loss' in r and 'epoch' in r:epoch_losses[r['epoch']].append(r['loss'])
            for ep,losses in epoch_losses.items():records.setdefault(ep,{'epoch':ep})['loss']=sum(losses)/len(losses)
            if epoch_losses:notes=['loss为最后一次尝试中日志采样平滑值的均值，不是完整epoch精确均值。']
        for p in sorted(base.rglob('epoch_*.json')):
            r=read(p)
            if isinstance(r,dict) and 'train' in r and 'loss' in r['train']:
                records.setdefault(r['epoch'],{'epoch':r['epoch']})['loss']=r['train']['loss'];notes=['loss为后端记录的整轮训练均值。']
        for p in sorted(base.rglob('results.csv')):
            with p.open(encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):
                    r={k.strip():float(v) for k,v in r.items()};ep=int(r['epoch'])
                    row=records.setdefault(ep,{'epoch':ep})
                    for k in ('train/box_loss','train/cls_loss','train/dfl_loss'):
                        if k in r:row[k]=r[k]
                    if 'metrics/mAP50-95(B)' in r:row.update(metric_values(r))
            notes=['YOLO分别记录box/cls/dfl损失；不将三者冒充统一加权总loss。']
        if records:groups[key+'/detector']={'rows':[records[k] for k in sorted(records)],'note':' '.join(notes),'primary':'mAP'}
        for p in sorted(base.rglob('*history.json')):
            hist=read(p)
            if not isinstance(hist,list) or not hist:continue
            branch='obb' if p.name=='obb_history.json' else p.parent.name if p.parent!=base else 'component'
            rows=[{k:v for k,v in r.items() if isinstance(v,(float,int)) and not isinstance(v,bool)} for r in hist]
            primary=next((k for k in ('mixed_score','val_mean_polygon_iou','mean_iou','leaf_top1') if any(k in r for r in rows)),None)
            groups[key+'/'+branch]={'rows':rows,'note':'loss为整轮训练均值；组件验证分不是完整检测分类流水线综合分。','primary':primary}
    return groups


def curves(dest,name,rows,primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (dest/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    for field in fields:
        if field=='epoch' or field in ('known_fine_targets','val_iou30_recall'):continue
        pts=[(r['epoch'],r[field]) for r in rows if field in r and r[field] is not None]
        if not pts:continue
        loss='loss' in field;ax=axes[0 if loss else 1]
        ax.plot(*zip(*pts),marker='o',label=field,linewidth=1.5)
    for ax,title in zip(axes,('Training loss','Validation metrics')):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        epochs=sorted({r['epoch'] for r in rows})
        if len(epochs)<=12:ax.set_xticks(epochs)
        if len(epochs)==1:ax.set_xlim(epochs[0]-.5,epochs[0]+.5)
        if ax.lines:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'Not recorded',ha='center',transform=ax.transAxes)
    fig.suptitle(name);fig.tight_layout();fig.savefig(dest/(name+'.png'),dpi=140);plt.close(fig)


def training_request(c):
    return dict(description='实际训练参数；修改此副本不改变已运行实验。新实验参数在竞赛配置中设置。',
                models=c.get('models'),dataset=c.get('dataset'),dataset_scale=c.get('full_dataset_scale'),
                command=c.get('argv'),resources=c.get('resources'),validation_interval=c.get('val_interval'),
                checkpoint_interval=c.get('save_interval'),started=c.get('started'),status=c.get('status'),
                scope='smoke subset' if '--smoke' in c.get('argv',[]) else 'full dataset')


def publish_training(c, plots=True, record=True):
    work=Path(c['run_dir']);public=Path(c.get('public_dir',public_dir(work)))
    initialize(public,training_request(c),c.get('status','unknown'))
    for model in c.get('models',[]):
        p=work/model/'training_config.json'
        if p.exists():
            conf=read(p);raw=conf.get('raw',conf)
            write(public/'config'/(model+'.json'),dict(actual_dataset_scale=conf.get('sizes'),**{k:raw[k] for k in ('model','train','data','runtime') if k in raw}))
        else:
            configs=[e for e in events(work) if e.get('kind')=='configuration' and e.get('model')==model]
            if configs:write(public/'config'/(model+'.json'),configs)
    groups=collect_training(work)
    result=['SARdetect 训练结果', '实验：'+public.name, '状态：'+c.get('status','unknown'),
            '执行尝试：'+str(c.get('attempt',1))+' 次（失败与续训输出保留在record.txt）',
            '数据：'+str(c.get('dataset','unknown')),
            '范围：'+('小样本功能测试，不代表完整数据成绩' if '--smoke' in c.get('argv',[]) else '完整训练配置'),
            '指标单位：分数为0～1；乘100即百分比。',
            '说明：下列最佳指标来自训练中的验证集，不使用test选轮。','']
    scale=c.get('full_dataset_scale',{})
    if scale:result+=['数据规模：'+' / '.join(f'{k}={v.get("images","?")}图' for k,v in scale.items()),'']
    epoch_text=[]
    for key,g in groups.items():
        rows=g['rows'];primary=g['primary'];valid=[r for r in rows if primary and r.get(primary) is not None and math.isfinite(r[primary])]
        best=max(valid,key=lambda r:r[primary]) if valid else None
        result+=['【'+key+'】',f'已记录轮次：{", ".join(str(r["epoch"]) for r in rows)}']
        if best:result+=['最佳验证轮次：'+str(best['epoch'])+'（按'+label(primary)+'选取）',numbers({k:v for k,v in best.items() if k not in ('epoch','loss') and 'loss' not in k})]
        else:result+=['最佳验证结果：尚未记录完整验证，不能报告成绩。']
        result+=[g['note'],'']
        epoch_text+=['【'+key+' 逐轮记录】',g['note']]+[numbers(r) for r in rows]+['']
        if plots:curves(public/'visualization',key.replace('/','-'),rows,primary)
    if not groups:result+=['尚无逐轮指标；查看record.txt中的初始化或失败信息。']
    if c.get('error'):result+=['失败/中断原因：'+c['error']]
    pipeline=work/'evaluations.json'
    if pipeline.exists():
        for name,e in read(pipeline).items():
            result+=['','【完整流水线评测：'+name+'】','split='+str(e['split']),numbers(e['metrics']), '评测目录：'+e['public_dir']]
    else:result+=['','完整流水线val/test：尚未执行或未关联。训练期mAP/组件得分不能当作竞赛综合分。']
    (public/'result.txt').write_text('\n'.join(result)+'\n',encoding='utf-8')
    if record:
        logs=sorted(work.glob('console-attempt-*.log'))
        if not logs:logs=sorted(set(work.rglob('console.log'))|set(work.rglob('train.log')))
        tmp=public/'record.txt.tmp'
        with tmp.open('w',encoding='utf-8') as f:
            f.write('SARdetect 训练记录\n实验：'+public.name+'\n状态：'+c.get('status','unknown')+'\n\n'+'\n'.join(epoch_text)+'\n原始训练输出（按尝试顺序，失败和中断保留）\n')
            for p in logs:
                f.write('\n--- '+str(p.relative_to(work))+' ---\n')
                with p.open(encoding='utf-8',errors='replace') as src:
                    for line in src:f.write(friendly(line))
        os.replace(tmp,public/'record.txt')
    (public/'visualization/README.txt').write_text('Training curves\nRun: '+public.name+'\nPNG: training losses and validation metrics per epoch. CSV: numeric values used in each plot. Missing validation remains missing; sampled MMEngine loss is not an exact full-epoch mean. See ../record.txt and ../result.txt.\n',encoding='utf-8')


def publish_evaluation(work,public=None,status=None):
    work=Path(work);public=Path(public or public_dir(work));req=read(work/'request.json')
    status=status or ('passed' if (work/'completed.json').exists() else 'failed')
    initialize(public,req,status)
    meta=req['dataset'];rows=read(work/'summary.json') if (work/'summary.json').exists() else []
    text=['SARdetect 评测/推理结果','状态：'+status,'实验：'+public.name,
          '数据集划分：'+str(meta.get('split') or '无标签'),f'图像数：{meta["images"]}；真值目标数：{meta["targets"]}',
          '范围：'+('功能抽样，不代表完整成绩' if meta.get('limited') else '完整输入'),
          '计分：旋转框IoU≥0.3，逐图平均；综合分=(检测分+分类分)/2。',
          '指标单位：分数为0～1；乘100即百分比。','']
    if not meta.get('labels_available'):text+=['无真值标签：仅输出预测，不计算或伪造分数。']
    elif meta.get('detection_only'):text+=['仅检查检测与OBB：分类分和综合分未计算。']
    for r in rows:text+=['【'+r['model']+'】',numbers(r),'']
    if (work/'failed.json').exists():text+=['失败详情：'+read(work/'failed.json')['error']]
    # Competition TXT is the public submission; JSON remains internal diagnostics.
    for row in rows:
        folder=work/row['model']/'labels'
        if folder.is_dir():
            dest=public/'config/predictions'/row['model']
            shutil.copytree(folder,dest,dirs_exist_ok=True)
            text+=['竞赛预测TXT：config/predictions/'+row['model']+'（类别 四点像素坐标 置信度）']
            if (work/row['model']/'labels-format.json').exists():
                shutil.copy2(work/row['model']/'labels-format.json',public/'config'/('labels-format-'+row['model']+'.json'))
    (public/'result.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    with (public/'record.txt').open('w',encoding='utf-8') as f:
        f.write('SARdetect 评测记录\n'+numbers({'elapsed_seconds':read(work/'completed.json').get('seconds',0)})+'\n' if (work/'completed.json').exists() else 'SARdetect 评测未完成\n')
        for p in sorted(work.rglob('*.log')):
            f.write('\n--- '+str(p.relative_to(work))+' ---\n')
            for line in p.read_text(encoding='utf-8',errors='replace').splitlines(True):f.write(friendly(line))
    publish_visualization(work,public)
    # Associate complete evaluation with the exact training run identified by checkpoint.
    if status=='passed':
        for r in rows:
            cp=req.get('assets',{}).get(r['model'],{}).get('detector',{}).get('path','')
            parts=Path(cp).parts
            marker=next((v for v in ('checkpoints','4.checkpoints') if v in parts),None)
            if not marker:continue
            name=parts[parts.index(marker)+1];train=ROOT/'.runtime/experiments/1.train'/name
            if not (train/'training.json').exists():continue
            p=train/'evaluations.json';saved=read(p) if p.exists() else {}
            saved[public.name]={'split':meta.get('split'),'metrics':r,'public_dir':str(public)};write(p,saved)
            context=read(train/'training.json')
            context.update(run_dir=str(train),public_dir=str(ROOT/'C.results/1.train'/name))
            publish_training(context,plots=False,record=False)


def publish_visualization(work,public):
    """Resampling must never overwrite the experiment's log or final metrics."""
    work=Path(work);public=Path(public);vis=public/'visualization'
    vis.mkdir(parents=True,exist_ok=True)
    for old in vis.glob('sample-*'):
        if old.is_file():old.unlink()
    for p in work.glob('*/examples/*'):
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):
            shutil.copy2(p,vis/('sample-'+p.parent.parent.name+'-'+p.name))
    (vis/'README.txt').write_text('Evaluation examples\nRun: '+public.name+'\nGreen polygons: ground truth. Red polygons: predictions. High/low: random samples within ranked score quartiles; unlabeled inputs are random without scores. See ../result.txt and config/run.json.\n',encoding='utf-8')
