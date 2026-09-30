"""Model-specific entry: entry.py train|test|infer [shared options]."""
import sys,os
from pathlib import Path
root=Path(__file__).resolve().parents[3]
action=sys.argv[1] if len(sys.argv)>1 else '--help'
os.execv(sys.executable,[sys.executable,str(root/'D.codes/2.utils/competition_cli.py'),action,'--model','model-3-s',*sys.argv[2:]])
