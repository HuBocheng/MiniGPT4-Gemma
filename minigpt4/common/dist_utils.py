"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import functools
import os

import torch
import torch.distributed as dist
import timm.models.hub as timm_hub


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    在非主进程中禁用打印操作
    在非主进程中重写print函数来实现的，在主进程（或者当明确指定force=True时）才允许打印信息。
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# 判断PyTorch分布式环境是否可用且已经初始化
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# 返回当前分布式环境中的总进程数
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# 返回当前进程的全局rank。这个rank是全局唯一的，用于标识分布式环境中的每个进程
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# 判断当前进程是否是主进程（rank=0）
def is_main_process():
    return get_rank() == 0


# 根据传入的参数初始化PyTorch的分布式环境
def init_distributed_mode(args):
    if args.distributed is False:
        print("Not using distributed mode")
        return
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            args.rank, args.world_size, args.dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# 获取当前进程的rank和总的world size，考虑了版本差异
def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


# 是一个装饰器，用于确保只有主进程（rank为0的进程）执行被装饰的函数
def main_process(func):

    # @functools.wraps()将原始函数（被装饰的函数）的一些重要属性（如名称、文档字符串、注解等）复制到装饰器内部的封装函数（通常称为包装器或wrapper）上
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # 任意数量的位置参数（*args）和关键字参数（**kwargs），使得它可以无缝地代理到任何被装饰的函数
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


# 从URL下载文件并将其缓存到本地
def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()
