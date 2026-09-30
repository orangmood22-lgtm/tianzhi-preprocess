#!/usr/bin/env python3
"""Split an images/labels competition dataset and write data.txt."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'D.codes/2.utils'))
from competition_split import main
if __name__=='__main__':main()
