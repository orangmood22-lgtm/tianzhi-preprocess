"""SIVED YOLOv8m warm-start training with the existing SIVED augmentation recipe."""
import argparse,os,sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[3]/'D.codes/2.utils'))
from sived_runtime import ROOT as S,read,write
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'D.codes/2.utils'))
from training_native import yolo_train
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();raw=read(a.config);tr=raw['train'];out=Path(raw['run']['output_root'])/raw['run']['id'];out.mkdir(parents=True,exist_ok=True)
 sys.path[:0]=[str(S/'D.codes/1.model/model5-YOLOv8m'),str(S/'D.codes/1.model/model5-YOLOv8m/support')]
 from runtime_paths import prepare_yolo_environment
 prepare_yolo_environment(out, S)
 import torch,torch_npu
 from ultralytics import YOLO
 from ultralytics.utils import SETTINGS
 from runtime import complete_nms_batches
 from csv_results import install_csv_reader
 install_csv_reader()
 from fonts import offline_font
 complete_nms_batches();offline_font();SETTINGS.update({'sync':False,'wandb':False,'mlflow':False,'comet':False,'clearml':False,'tensorboard':False})
 kwargs=dict(data=raw['data']['data_yaml'],epochs=tr['epochs'],patience=8,imgsz=1024,batch=tr['batch_size'],nbs=16,workers=tr['workers'],amp=True,optimizer='SGD',lr0=tr['learning_rate'],lrf=.05,momentum=.937,weight_decay=.0005,warmup_epochs=1,seed=raw['runtime']['seed'],deterministic=True,single_cls=True,box=10.,cls=.5,dfl=1.5,mosaic=.25,close_mosaic=5,hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=.5,mixup=0,copy_paste=0,auto_augment=None,erasing=0,nms=True,conf=.001,iou=.7,max_det=300,plots=False,save=True,save_period=-1,val=True,cache=False,device='npu:0',project=str(out),name='fit',exist_ok=False,augmentations=[])
 kwargs.update(imgsz=int(raw['model']['input_size'][0]),patience=tr.get('patience',8),amp=tr.get('amp',True),optimizer=tr.get('optimizer','SGD'),nbs=tr.get('nominal_batch_size',64),lrf=tr['final_lr_factor'],momentum=tr.get('momentum',.937),weight_decay=tr.get('weight_decay',.0005),warmup_epochs=tr['warmup_epochs'],mosaic=tr.get('mosaic',.25),close_mosaic=tr['close_mosaic'],box=tr.get('box_loss_gain',10.),cls=tr.get('class_loss_gain',.5),dfl=tr.get('dfl_loss_gain',1.5))
 if tr.get('finetune'):kwargs['warmup_bias_lr']=tr['warmup_bias_lr']
 if a.smoke:kwargs.update(close_mosaic=0)
 write(out/'train_kwargs.json',kwargs);yolo_train(raw,kwargs,YOLO)
if __name__=='__main__':main()
