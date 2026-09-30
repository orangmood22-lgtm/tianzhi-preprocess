#!/usr/bin/env python3
"""Offline Linux/Ascend resource monitor. Python 3.6+ standard library only."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def percent(used, total):
    return 100.0 * used / total if used is not None and total and total > 0 else None


def cpu_times(text):
    # guest/guest_nice are already included in user/nice; do not count twice.
    for line in (text or '').splitlines():
        if line.startswith('cpu '):
            values = [int(x) for x in line.split()[1:9]]
            return sum(values), values[3] + values[4]
    return None


def cpu_percent(before, after):
    if before is None or after is None:
        return None
    total, idle = after[0] - before[0], after[1] - before[1]
    if total <= 0 or idle < 0:
        return None
    return max(0.0, min(100.0, 100.0 * (total - idle) / total))


def host_memory():
    values = {}
    for line in (read('/proc/meminfo') or '').splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    total, available = values.get('MemTotal'), values.get('MemAvailable')
    # Do not mistake MemFree for available memory when MemAvailable is absent.
    used = total - available if total is not None and available is not None else None
    swap_total = values.get('SwapTotal')
    swap_used = (swap_total - values['SwapFree']
                 if swap_total is not None and 'SwapFree' in values else None)
    return dict(total_bytes=total, available_bytes=available, used_bytes=used,
                used_percent=percent(used, total), swap_total_bytes=swap_total,
                swap_used_bytes=swap_used)


def unescape_mount(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), value)


def locate_cgroup(controller, memberships, mounts):
    """Resolve both host mounts and mounts rooted at a container's cgroup."""
    groups = []
    for line in memberships.splitlines():
        _, controllers, path = line.split(':', 2)
        if controller in controllers.split(',') or not controllers:
            groups.append((2 if not controllers else 1, path))
    # Prefer the controller's v1 group in hybrid systems.
    for version, path in sorted(groups):
        for line in mounts.splitlines():
            fields, fs = line.split(' - ', 1)
            fields, fs = fields.split(), fs.split()
            if fs[0] != ('cgroup2' if version == 2 else 'cgroup'):
                continue
            if version == 1 and controller not in fs[2].split(','):
                continue
            root, mount = unescape_mount(fields[3]), Path(unescape_mount(fields[4]))
            if path == root:
                relative = ''
            elif path.startswith(root.rstrip('/') + '/'):
                relative = path[len(root.rstrip('/')):].lstrip('/')
            else:
                continue
            if '..' in Path(relative).parts:
                continue
            return version, mount / relative, mount
    return None


def ancestors(location):
    if location:
        _, current, mount = location
        while True:
            yield current
            if current == mount:
                break
            current = current.parent


def memory_cgroups(location):
    rows = []
    if not location:
        return rows
    version = location[0]
    limit_name = 'memory.max' if version == 2 else 'memory.limit_in_bytes'
    usage_name = 'memory.current' if version == 2 else 'memory.usage_in_bytes'
    for index, path in enumerate(ancestors(location)):
        raw = read(path / limit_name)
        limit = number(raw)
        if limit is not None and limit >= 2 ** 60:
            limit = None  # v1 unlimited sentinel
        usage = number(read(path / usage_name))
        if raw is None and usage is None:
            continue
        # Include current cgroup even if unlimited, plus finite ancestor limits.
        if index == 0 or limit is not None:
            rows.append(dict(scope='current' if index == 0 else 'ancestor',
                             path=str(path), used_bytes=usage, limit_bytes=limit,
                             limit_raw=raw, used_percent=percent(usage, limit)))
    return rows


def cgroup_cpu_seconds(location):
    if not location:
        return None
    version, path, _ = location
    if version == 1:
        value = number(read(path / 'cpuacct.usage'))
        return value / 1e9 if value is not None else None
    for line in (read(path / 'cpu.stat') or '').splitlines():
        key, value = line.split()
        if key == 'usage_usec':
            return int(value) / 1e6
    return None


def cpu_quotas(location):
    result = []
    if location:
        for path in ancestors(location):
            if location[0] == 2:
                values = (read(path / 'cpu.max') or '').split()
                quota, period = (map(number, values) if len(values) == 2 else (None, None))
            else:
                quota = number(read(path / 'cpu.cfs_quota_us'))
                period = number(read(path / 'cpu.cfs_period_us'))
            if quota is not None and quota > 0 and period and period > 0:
                result.append(dict(path=str(path), cores=quota / period))
    return result


def parse_npu(text):
    """Parse paired device/chip rows from the 910B npu-smi info table."""
    devices, pending = [], None
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        columns = [part.strip() for part in line.split('|')[1:-1]]
        if len(columns) != 3:
            continue
        card = re.fullmatch(r'(\d+)\s+(\S+)', columns[0])
        top = columns[2].split()
        if card and len(top) >= 2 and re.fullmatch(r'\d+', columns[1]) is None:
            pending = dict(id=int(card.group(1)), name=card.group(2),
                           health=columns[1], power_w=number(top[0]),
                           temperature_c=number(top[1]))
            continue
        if pending and re.fullmatch(r'\d+', columns[0]):
            stats = re.fullmatch(r'(\S+)\s+(\S+)\s*/\s*(\S+)\s+(\S+)\s*/\s*(\S+)', columns[2])
            if not stats:
                pending = None
                continue
            core, _, _, used, total = stats.groups()
            used, total = number(used), number(total)
            pending.update(chip=int(columns[0]), bus_id=columns[1],
                           aicore_percent=number(core.rstrip('%')),
                           hbm_used_mb=used, hbm_total_mb=total,
                           hbm_used_percent=percent(used, total))
            devices.append(pending)
            pending = None
    return devices


def get_npu(timeout, disabled=False):
    if disabled:
        return dict(devices=[], error='NPU query disabled (--no-npu).', raw='')
    command = shutil.which('npu-smi')
    if not command:
        return dict(devices=[], error='npu-smi not found; CPU/memory/disk still available.', raw='')
    try:
        proc = subprocess.run([command, 'info'], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True,
                              errors='replace', timeout=timeout,
                              env=dict(os.environ, LC_ALL='C'))
        raw = proc.stdout.strip()
        devices = parse_npu(raw) if proc.returncode == 0 else []
        error = None
        if proc.returncode:
            error = 'npu-smi exit {}: {}'.format(proc.returncode, proc.stderr.strip()[:500])
        elif not devices:
            error = 'Unrecognized npu-smi table or no visible NPU; inspect raw output.'
        return dict(devices=devices, error=error, raw=raw)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(devices=[], error=str(exc), raw='')


def disks(paths):
    rows = []
    for path in dict.fromkeys(paths):
        try:
            stat = os.statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            used = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
            available = stat.f_bavail * stat.f_frsize
            rows.append(dict(path=path, total_bytes=total, used_bytes=used,
                             available_bytes=available, used_percent=percent(used, used + available)))
        except OSError as exc:
            rows.append(dict(path=path, error=str(exc)))
    return rows


def sample(args):
    membership = read('/proc/self/cgroup') or ''
    mounts = read('/proc/self/mountinfo') or ''
    mem_loc = locate_cgroup('memory', membership, mounts)
    quota_loc = locate_cgroup('cpu', membership, mounts)
    usage_loc = locate_cgroup('cpuacct', membership, mounts)
    start = time.monotonic()
    before = cpu_times(read('/proc/stat'))
    cg_before = cgroup_cpu_seconds(usage_loc)
    time.sleep(args.sample)
    after = cpu_times(read('/proc/stat'))
    cg_after = cgroup_cpu_seconds(usage_loc)
    elapsed = time.monotonic() - start
    cg_cores = ((cg_after - cg_before) / elapsed
                if cg_after is not None and cg_before is not None and cg_after >= cg_before else None)
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return dict(timestamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                hostname=platform.node(), architecture=platform.machine(),
                cpu=dict(host_busy_percent=cpu_percent(before, after), logical_cpus=os.cpu_count(),
                         affinity_cpus=affinity, sample_seconds=elapsed,
                         cgroup_used_cores=cg_cores, visible_quotas=cpu_quotas(quota_loc)),
                load_average=list(os.getloadavg()), memory=host_memory(),
                cgroup_memory=memory_cgroups(mem_loc), disks=disks(args.disk),
                npu=get_npu(args.npu_timeout, args.no_npu))


def fmt(value, suffix=''):
    return 'N/A' if value is None else '{:.1f}{}'.format(value, suffix)


def gib(value):
    return fmt(value / 1024 ** 3 if value is not None else None, ' GiB')


def display(data, raw_npu=False):
    cpu, mem = data['cpu'], data['memory']
    print('SARdetect 服务器状态 | {} | {} ({})'.format(data['timestamp'], data['hostname'], data['architecture']))
    print('CPU /proc: {} | 逻辑CPU {} | 当前亲和性 {} | 采样 {:.2f}s'.format(
        fmt(cpu['host_busy_percent'], '%'), cpu['logical_cpus'], cpu['affinity_cpus'], cpu['sample_seconds']))
    print('系统负载 1/5/15分钟: ' + ' / '.join(fmt(x) for x in data['load_average']))
    print('主存 /proc: 已用 {} / 总量 {} ({}) | 可用 {}'.format(
        gib(mem['used_bytes']), gib(mem['total_bytes']), fmt(mem['used_percent'], '%'), gib(mem['available_bytes'])))
    print('Swap: {} / {}'.format(gib(mem['swap_used_bytes']), gib(mem['swap_total_bytes'])))
    print('当前cgroup CPU: {} 核当量（1.0 = 占满1个逻辑CPU）'.format(fmt(cpu['cgroup_used_cores'])))
    for quota in cpu['visible_quotas']:
        print('  CPU配额: {} 核 | {}'.format(fmt(quota['cores']), quota['path']))
    if not cpu['visible_quotas']:
        print('  未读到有限CPU配额；仍受亲和性及不可见上级约束。')
    if not data['cgroup_memory']:
        print('cgroup内存: N/A（控制器不可见或未提供统计）')
    for row in data['cgroup_memory']:
        cap = gib(row['limit_bytes']) if row['limit_bytes'] is not None else '未读到有限上限'
        print('cgroup内存 [{}]: {} / {} ({}) | {}'.format(
            row['scope'], gib(row['used_bytes']), cap, fmt(row['used_percent'], '%'), row['path']))
    print('\n磁盘（所在文件系统，不是目录大小）:')
    for row in data['disks']:
        if 'error' in row:
            print('  {}: {}'.format(row['path'], row['error']))
        else:
            print('  {}: {} / {} ({}) | 可用 {}'.format(
                row['path'], gib(row['used_bytes']), gib(row['total_bytes']),
                fmt(row['used_percent'], '%'), gib(row['available_bytes'])))
    print('\nNPU / Chip  型号       健康   AI Core   HBM显存(MB)        显存占用   温度     功耗')
    for row in data['npu']['devices']:
        print('  {id}/{chip}       {name:<10} {health:<6} {core:>7}  {used:>7}/{total:<7} {mem:>8} {temp:>8} {power:>8}'.format(
            core=fmt(row['aicore_percent'], '%'), used=fmt(row['hbm_used_mb']),
            total=fmt(row['hbm_total_mb']), mem=fmt(row['hbm_used_percent'], '%'),
            temp=fmt(row['temperature_c'], ' C'), power=fmt(row['power_w'], ' W'), **row))
    if data['npu']['error']:
        print('  提示: ' + data['npu']['error'])
    if raw_npu or (data['npu']['error'] and data['npu']['raw']):
        print(data['npu']['raw'])
    print('\n口径: /proc通常是宿主机统计；cgroup为当前组/可见上级，内存计数含缓存。')
    print('NPU显示驱动可见设备；显存占用不等于计算利用率。监控不加载模型、不修改配置。')


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return result


def main():
    parser = argparse.ArgumentParser(description='Linux/Ascend离线资源检查：仅使用Python标准库与现有npu-smi。')
    parser.add_argument('--watch', action='store_true', help='持续刷新，Ctrl+C退出')
    parser.add_argument('--interval', type=positive, default=2.0, help='刷新目标周期/秒，默认2')
    parser.add_argument('--sample', type=positive, default=0.5, help='CPU采样窗口/秒，默认0.5')
    parser.add_argument('--count', type=int, default=0, help='watch输出次数，0表示无限')
    parser.add_argument('--json', action='store_true', help='输出JSON；watch时每行一个JSON')
    parser.add_argument('--raw-npu', action='store_true', help='同时输出npu-smi原始表及进程列表')
    parser.add_argument('--no-npu', action='store_true', help='跳过NPU查询')
    parser.add_argument('--npu-timeout', type=positive, default=5.0, help='NPU查询超时/秒')
    parser.add_argument('--disk', action='append', help='检查所在文件系统，可重复指定')
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('此程序需在Linux服务器运行。')
    if args.count < 0:
        parser.error('--count must be >= 0')
    if args.disk is None:
        args.disk = ['/', str(Path(__file__).resolve().parent), '/dev/shm']
    count = 0
    try:
        while True:
            start = time.monotonic()
            data = sample(args)
            if args.json:
                if not args.raw_npu and not data['npu']['error']:
                    data['npu'].pop('raw', None)
                print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)
            else:
                if args.watch and sys.stdout.isatty() and os.environ.get('TERM') != 'dumb':
                    print('\033[2J\033[H', end='')
                display(data, args.raw_npu)
                sys.stdout.flush()
            count += 1
            if not args.watch or (args.count and count >= args.count):
                return 0
            time.sleep(max(0, args.interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:
        return 0
    except (OSError, ValueError) as exc:
        print('无法读取系统统计: {}'.format(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
