"""Shared configuration, logging, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def resolve_path(value: Optional[str], base: Optional[Path] = None, required: bool = False) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError("A required path is empty")
        return None
    if "REPLACE_" in str(value):
        raise ValueError(f"Configuration still contains a placeholder: {value}")
    path = Path(os.path.expandvars(str(value))).expanduser()
    old_prefix = '/workspace/SARdetect/'
    if str(value).startswith(old_prefix):
        path = PROJECT_ROOT / str(value)[len(old_prefix):]
    if not path.is_absolute():
        path = (base or PROJECT_ROOT) / path
    path = path.resolve()
    if required and not path.exists():
        raise FileNotFoundError(path)
    return path


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def environment_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
    }
    try:
        import torch_npu

        snapshot["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
    except ImportError:
        snapshot["torch_npu"] = None
    try:
        import mmcv
        import mmdet
        import mmengine

        snapshot.update(mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmengine=mmengine.__version__)
    except ImportError:
        pass
    return snapshot


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # torchrun starts every local rank at once.  A fixed .tmp filename lets
    # one rank replace another rank's temporary file before it can do so.
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def prepare_run_directory(
    output_root: Path,
    run_id: str,
    config: Mapping[str, Any],
    command: Optional[str] = None,
) -> Path:
    if not run_id or any(char in run_id for char in "/\\"):
        raise ValueError("run_id must be a non-empty directory name")
    run_dir = (output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.resolved.json", dict(config))
    write_json(run_dir / "environment.json", environment_snapshot())
    (run_dir / "command.txt").write_text(command or " ".join(sys.argv), encoding="utf-8")
    return run_dir


def configure_logging(run_dir: Path, is_main_process: bool = True) -> logging.Logger:
    logger = logging.getLogger("sar_train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO if is_main_process else logging.WARNING)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if is_main_process:
        file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
