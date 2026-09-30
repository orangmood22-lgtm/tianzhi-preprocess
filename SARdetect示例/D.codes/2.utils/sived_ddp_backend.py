"""HCCL wrapper preserving the registered OpenMMLab model architecture."""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import openmmlab_backend as backend
rank=int(os.environ.get('RANK','0'));original_build=backend.build_runner_cfg;original_prepare=backend.prepare_run_directory;original_write=backend.write_json
def build(raw,run):
 cfg,provenance=original_build(raw,run);cfg['launcher']='pytorch';cfg['model_wrapper_cfg']=dict(type='MMDistributedDataParallel',find_unused_parameters=True,broadcast_buffers=False)
 bs=int(raw['train'].get('per_device_batch',1));cfg['train_dataloader']['batch_size']=bs
 for name in ['train_dataloader','val_dataloader','test_dataloader']:
  cfg[name]['num_workers']=int(raw['train'].get('workers',0));cfg[name]['persistent_workers']=cfg[name]['num_workers']>0
 provenance.update(distributed_world_size=int(os.environ['WORLD_SIZE']),per_device_batch=bs)
 return cfg,provenance
def prepare(root,rid,raw):
 if rank==0:return original_prepare(root,rid,raw)
 p=(root/rid).resolve();p.mkdir(parents=True,exist_ok=True);return p
def write(p,d):
 if rank==0:original_write(p,d)
backend.build_runner_cfg=build;backend.prepare_run_directory=prepare;backend.write_json=write
backend.main()
