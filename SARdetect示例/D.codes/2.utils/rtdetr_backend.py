"""Compatibility entry; implementation belongs to model3-RT-DETRv2."""
from pathlib import Path
import sys,runpy
_p=Path(__file__).resolve().parents[1]/'1.model/model3-RT-DETRv2/rtdetr_backend.py'
sys.path.insert(0,str(_p.parent))
if __name__=='__main__':runpy.run_path(str(_p),run_name='__main__')
else:globals().update(runpy.run_path(str(_p)))
