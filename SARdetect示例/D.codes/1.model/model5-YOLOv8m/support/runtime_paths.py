"""Keep Ultralytics settings inside the owning run, independent of global caches."""
import json
import os
from pathlib import Path


def prepare_yolo_environment(run_dir, project_root):
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    config_root = run_dir / 'settings'
    # This vendored version appends Ultralytics and checks the parent first.
    # A missing parent silently selects /tmp/Ultralytics and its stale settings.
    config_dir = config_root / 'Ultralytics'
    config_dir.mkdir(parents=True, exist_ok=True)
    settings_file = config_dir / 'settings.json'
    settings = json.loads(settings_file.read_text(encoding='utf-8')) if settings_file.exists() else {}
    settings.update(datasets_dir=str(project_root / 'A.dataset'),
                    weights_dir=str(project_root / 'B.models'),
                    runs_dir=str(project_root / 'C.results'),
                    sync=False, wandb=False, mlflow=False, comet=False,
                    clearml=False, tensorboard=False, dvc=False, raytune=False)
    settings_file.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    os.environ.update(YOLO_CONFIG_DIR=str(config_root), YOLO_AUTOINSTALL='false',
                      YOLO_OFFLINE='true', MPLCONFIGDIR=str(run_dir / 'mpl'))
    return settings_file
