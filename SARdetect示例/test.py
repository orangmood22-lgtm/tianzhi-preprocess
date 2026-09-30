from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'D.codes/2.utils'))
from evaluation_runtime import main
if __name__ == '__main__':
    main('test')
