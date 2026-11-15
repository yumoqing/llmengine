"""推理服务的自保工具（无需 root，等价于 systemd drop-in 的兜底）。

背景（2026-11-14 生产事故）：embedding/rerank 服务每请求新建线程池，torch 在
每个新工作线程里各拉一套 libgomp 线程池且不回收 → 1 小时 230 次请求后进程
2804 线程 / 12.7GB RSS，空闲仍自旋吃 2.5~4.5 核（系统态 CPU 70%），把同机
doit 应用从 1.4ms 拖到 5s，MariaDB 也一起变慢。日志里还混进 30KB 的工具清单。

本模块提供两件事：
1. set_default_env()：在 import torch 之前把 OMP/MKL/OPENBLAS 线程数与自旋策略
   收敛（libgomp 在建线程池时读环境变量，晚了就没用，所以必须在入口最前面调）。
2. start_guard()：后台看门狗，定期检查自身 RSS 与线程数，超阈值就 exit(1)，
   让 systemd（Restart=on-failure/always）重启自己——宁可重启一个推理服务，
   也不要拖死整台机器。
"""
import os
import sys


def set_default_env():
    """在 import torch 之前调用。已存在的环境变量不覆盖（systemd 可覆盖默认值）。"""
    defaults = {
        'OMP_NUM_THREADS': '4',
        'OMP_WAIT_POLICY': 'PASSIVE',
        'MKL_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '4',
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def _rss_mb(pid=None):
    pid = pid or os.getpid()
    try:
        with open('/proc/%d/status' % pid, 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _thread_count(pid=None):
    pid = pid or os.getpid()
    try:
        return len(os.listdir('/proc/%d/task' % pid))
    except Exception:
        return 0


def start_guard(name, max_rss_mb=None, max_threads=None, interval=None):
    """启动看门狗线程；超阈值 exit(1) 交给 systemd 重启。失败绝不抛出。"""
    try:
        max_rss_mb = float(max_rss_mb or os.environ.get('GUARD_MAX_RSS_MB') or 6144)
        max_threads = int(max_threads or os.environ.get('GUARD_MAX_THREADS') or 300)
        interval = float(interval or os.environ.get('GUARD_INTERVAL_S') or 120)
    except ValueError:
        max_rss_mb, max_threads, interval = 6144.0, 300, 120.0
    try:
        import threading
        import time
        from appPublic.log import debug, exception

        def _loop():
            while True:
                time.sleep(interval)
                try:
                    rss = _rss_mb()
                    thr = _thread_count()
                    if rss > max_rss_mb or thr > max_threads:
                        exception('%s guard tripped: rss=%.0fMB (limit %.0f) threads=%d (limit %d) -> exit(1) for systemd restart'
                                  % (name, rss, max_rss_mb, thr, max_threads))
                        sys.stderr.flush()
                        os._exit(1)
                except Exception:
                    pass

        t = threading.Thread(target=_loop, name='%s-guard' % name, daemon=True)
        t.start()
        try:
            from appPublic.log import debug
            debug('%s guard started: max_rss=%.0fMB max_threads=%d interval=%.0fs'
                  % (name, max_rss_mb, max_threads, interval))
        except Exception:
            pass
        return True
    except Exception:
        return False