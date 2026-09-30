"""Unified registered SARdetect train/test/infer command router."""
import argparse,json,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from training_resources import interpreter
PY=interpreter()
def main():
 if len(sys.argv)>1 and sys.argv[1] in ['test','infer','pipeline']:
  from evaluation_runtime import main as evaluate
  return evaluate('infer' if sys.argv[1]=='infer' else 'test',sys.argv[2:])
 p=argparse.ArgumentParser(description=__doc__,add_help=False)
 p.add_argument('action',choices=['train','test','infer','pipeline']);p.add_argument('--model');p.add_argument('--models',nargs='+');p.add_argument('--dataset');p.add_argument('--checkpoint',type=Path);p.add_argument('--source',type=Path)
 a,rest=p.parse_known_args()
 if a.action=='train' and ('--help' in rest or '-h' in rest):
  print('SARdetect train.py --model MODEL --dataset DATASET [--epochs N] [--batch-size N] [--learning-rate LR] [--seed N] [--device npu:0] [--val-interval N] [--save-interval N] [--log-interval N] [--smoke] [--run-id DATASET_modelN_cfg00_YYYYMMDDHHMMSS]\nResume: train.py --resume C.results/1.train/RUN\nOptional classifier gate: --depth-prior P1 P2 P3; --branch all|baseline|aligned|aligned3924|convnext|dino.\nIntervals default to 1 epoch for validation/checkpoints and 20 steps for progress. Last epoch always validates. --smoke limits data and adjusts YOLO accumulation/initial AMP scale for a short functional check; --epochs remains effective.')
  return
 if a.action=='train' and '--resume' in rest:
  from training_control import supervise,read
  from experiment_output import resolve_run
  resume=resolve_run(rest[rest.index('--resume')+1])
  saved=read(resume/'training.json')
  requested=a.models or ([a.model] if a.model else [])
  requested=['model-'+n if not n.startswith('model-') else n for n in requested]
  if requested and requested!=saved['models']:raise ValueError('Resume model differs from saved run')
  if a.dataset and (ROOT/'A.dataset'/a.dataset).resolve()!=Path(saved['dataset']):raise ValueError('Resume dataset differs from saved run')
  if a.checkpoint or a.source:raise ValueError('Resume uses the saved checkpoint and data')
  return supervise('',rest)
 catalog=json.loads((ROOT/'D.codes/1.model/catalog.json').read_text())
 names=a.models or [a.model or 'model-1-s'];names=['model-'+n if not n.startswith('model-') else n for n in names]
 if any(n not in catalog for n in names):raise ValueError('Available models: '+', '.join(catalog))
 kinds={'legacy' if n in ['model-c','model-d'] else 'v3' if n.endswith('-s') else 'sived' for n in names}
 if len(kinds)!=1:raise ValueError('Select models from one dataset family per command')
 script={'legacy':'legacy_components_cli.py','v3':'cascade_cli.py','sived':'sived_runtime.py'}[kinds.pop()]
 if not a.dataset and len({catalog[n]['dataset'] for n in names})!=1:raise ValueError('Specify --dataset')
 if a.checkpoint:
  cp=a.checkpoint.resolve(strict=True)
  if len(names)!=1:raise ValueError('--checkpoint requires one model')
  if cp.suffix=='.json':
   candidate=json.loads(cp.read_text())
   if candidate.get('status') in ['invalid','failed']:raise ValueError('Candidate failed validation: '+candidate.get('reason',str(cp)))
   if candidate.get('model') and candidate['model']!=names[0]:raise ValueError('Candidate belongs to '+candidate['model']+', not '+names[0])
   if 'classifier' not in candidate and 'obb' not in candidate:
    if 'manifest' in candidate:target=candidate['manifest'];digest=candidate.get('sha256')
    elif 'checkpoints' in candidate:target=candidate['checkpoints'][0]['path'];digest=candidate['checkpoints'][0].get('sha256')
    elif 'branches' in candidate and 'refiner' in candidate['branches']:target=candidate['branches']['refiner']['checkpoint'];digest=candidate['branches']['refiner'].get('sha256')
    else:raise ValueError('Unknown candidate JSON format')
    cp=Path(target);cp=(ROOT/cp).resolve(strict=True)
    if digest:
     h=hashlib.sha256()
     with cp.open('rb') as f:
      for block in iter(lambda:f.read(2**20),b''):h.update(block)
     if h.hexdigest()!=digest:raise ValueError('Candidate hash mismatch: '+str(cp))
  rest+=['--checkpoint',str(cp)]
 if a.source:
  if a.action!='infer' or any(catalog[n]['role']!='detector' for n in names):raise ValueError('--source is for detector/pipeline inference; components require target proposals')
  from inference_input import prepare
  split=rest[rest.index('--split')+1] if '--split' in rest else 'test'
  a.dataset=str(prepare(a.source,split));os.environ['SARDETECT_UNLABELED']='1'
 args=[a.action,'--models',*names,'--dataset',a.dataset or catalog[names[0]]['dataset'],*rest]
 if a.action=='train':
  from training_control import supervise
  return supervise(script,args)
 from training_control import valid_name
 if '--run-id' in rest and not valid_name(rest[rest.index('--run-id')+1]):raise ValueError('Use <dataset>_modelN_cfg00_YYYYMMDD[HHMMSS] for --run-id')
 if '--output' in rest:
  output=Path(rest[rest.index('--output')+1]).resolve()
  if output.is_relative_to(ROOT/'C.results') and not valid_name(output.name):raise ValueError('C.results output requires a standard run name')
 if '--run-id' not in rest and '--output' not in rest:
  from training_control import naming
  args+=['--run-id',naming(a.source.stem if a.source else a.dataset or catalog[names[0]]['dataset'],names)]
 os.execv(PY,[PY,'-u',str(ROOT/'D.codes/2.utils'/script),*args])
if __name__=='__main__':main()
