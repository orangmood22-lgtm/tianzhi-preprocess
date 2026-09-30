"""Bounded host-memory resources and portable interpreter selection."""
import os
import sys
from collections import OrderedDict
from pathlib import Path


def release_host_memory():
    """Return freed CPU checkpoint buffers to glibc; never changes live tensors."""
    import gc
    import ctypes
    gc.collect()
    if sys.platform.startswith('linux'):
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except AttributeError:
            pass


def interpreter(yolo=False):
    explicit = os.environ.get('SARDETECT_YOLO_PYTHON' if yolo else 'SARDETECT_PYTHON')
    if explicit:
        return explicit
    return sys.executable


class ImageCache:
    """Lazy grayscale images, byte-bounded; oversized images are never cached."""
    def __init__(self, paths, megabytes=128):
        self.paths = paths
        self.limit = megabytes * 1024**2
        self.cache = OrderedDict()
        self.bytes = 0

    def __getitem__(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        import cv2
        value = cv2.imread(str(self.paths[key]), 0)
        if value is None:
            raise ValueError('Unreadable image: ' + str(self.paths[key]))
        while self.cache and self.bytes + value.nbytes > self.limit:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.nbytes
        if value.nbytes <= self.limit:
            self.cache[key] = value
            self.bytes += value.nbytes
        return value


def memory_available():
    """Available memory respects cgroup v1/v2 container limits when present."""
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        values[name] = int(value.strip().split()[0]) * 1024
    total, available = values['MemTotal'], values['MemAvailable']
    candidates = [(Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')),
                  (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'), Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'))]
    # Include non-namespace cgroup mounts (e.g. systemd on the competition host).
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, group = line.split(':', 2)
        if '..' in Path(group).parts:
            continue
        if controllers == '':
            base = Path('/sys/fs/cgroup') / group.lstrip('/')
            candidates.append((base/'memory.max', base/'memory.current'))
        elif 'memory' in controllers.split(','):
            base = Path('/sys/fs/cgroup/memory') / group.lstrip('/')
            candidates.append((base/'memory.limit_in_bytes', base/'memory.usage_in_bytes'))
    for limit, used in candidates:
        if limit.is_file() and used.is_file():
            text = limit.read_text().strip()
            if text != 'max':
                cap = int(text)
                total = min(total, cap)
                available = min(available, max(0, cap - int(used.read_text())))
    return total, available


def process_tree_rss(root):
    rows = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            row = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
            rows[int(path.parent.name)] = (int(row['PPid']), int(row.get('VmRSS', '0 kB').split()[0]) * 1024)
        except (OSError, ValueError, KeyError):
            pass
    family = {root}
    while True:
        children = {pid for pid, (parent, _) in rows.items() if parent in family}
        if children <= family:
            break
        family |= children
    return sum(rows.get(pid, (0, 0))[1] for pid in family)
