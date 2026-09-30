"""Load model-owned Python packages without adding model folders to sys.path.

The catalog keeps user-facing folders such as modelcs-HierarchicalEnsemble.
Python package aliases are private and stable; intra-model imports are relative.
"""
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[2]
_LOCK = RLock()

def load_model_module(model_id, module):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', module):
        raise ValueError('Expected one model module name')
    catalog = json.loads((ROOT/'D.codes/1.model/catalog.json').read_text(encoding='utf-8'))
    entry = catalog[model_id]
    folder = (ROOT/entry['directory']).resolve(strict=True)
    if not folder.is_relative_to((ROOT/'D.codes/1.model').resolve()):
        raise ValueError('Model package outside D.codes/1.model')
    package = '_sardetect_' + model_id.replace('-', '_')
    with _LOCK:
        if package not in sys.modules:
            spec = importlib.util.spec_from_file_location(package, folder/'__init__.py', submodule_search_locations=[str(folder)])
            obj = importlib.util.module_from_spec(spec)
            sys.modules[package] = obj
            try: spec.loader.exec_module(obj)
            except BaseException:
                sys.modules.pop(package, None)
                raise
        return importlib.import_module(package + '.' + module)
