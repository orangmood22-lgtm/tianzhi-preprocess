# 服务器状态检查

适用 Linux ARM64/x86_64、昇腾910B服务器。无需联网或安装Python包；仅使用Python 3.6+标准库、Linux `/proc`/cgroup和已有的`npu-smi`。启动脚本使用`-S`禁用site-packages，不依赖Conda、PyTorch或psutil。不加载模型、不申请NPU张量、不修改服务器配置。

在SARdetect目录运行：

```bash
bash server_status.sh                         # 查看一次
bash server_status.sh --watch                 # 每2秒刷新，Ctrl+C退出
bash server_status.sh --watch --interval 5    # 每5秒刷新
bash server_status.sh --raw-npu               # 附带npu-smi原始表和进程列表
bash server_status.sh --json                  # 机器可读JSON
bash server_status.sh --watch --count 3 --json # 输出3行JSON后退出
bash server_status.sh --disk /workspace --disk /dev/shm
```

也可直接执行`python3 -B -S server_status.py`。从其他目录用绝对路径调用sh即可。

显示CPU采样利用率、CPU亲和性、1/5/15分钟负载、主存/可用内存、Swap、cgroup当前内存及可见上级限额、cgroup消耗CPU核当量与配额、磁盘容量/剩余空间，以及每张NPU的HBM显存占用率、AI Core利用率、健康状态、温度和功耗。

口径与限制：

- CPU默认采样0.5秒；系统利用率范围0–100%，不重复计入guest时间。cgroup的1.0核当量表示占满一个逻辑CPU，多个核可大于1。
- `/proc`通常反映宿主机；容器限额另列。cgroup内存包含缓存，与主存的`总量-MemAvailable`口径不同。祖先行使用祖先自己的占用量，不将兄弟容器占用漏算。隐藏的上级限额无法读取；没有读到有限限额不代表资源无限。
- 磁盘显示路径所在文件系统容量，不递归计算SARdetect目录大小。同一文件系统的不同路径可能显示相同数值。
- NPU显示驱动可见设备，可能多于比赛分配设备；只读查询不会使用这些设备训练。HBM的MB单位沿用npu-smi，CPU内存/磁盘以GiB显示。
- 已适配当前910B3、25.5.1的表格；不同驱动若格式无法识别会提示并保留原始输出，不将未知值当成零。无`npu-smi`或查询失败时其他指标仍可用；超时默认5秒。刷新周期受查询耗时限制。
- JSON的缺失值为`null`，无NPU/驱动错误见`npu.error`；这不是所有资源健康状态的自动判定。`--no-npu`可跳过驱动查询。

内核口径参考：https://docs.kernel.org/filesystems/proc.html 、https://docs.kernel.org/admin-guide/cgroup-v2.html 。
