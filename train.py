"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import logging


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now,view_gpu,view_model_device_allocation,check_frozen_parts

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


# 确定伪随机的随机数种子
def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    logging.info("before clean, free memory: {}".format(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)))
    
    
#     # 创建一个小张量
#     tensor_cpu = torch.tensor([1, 2, 3, 4, 5])

#     # 将张量移动到GPU上
#     tensor_gpu = tensor_cpu.to('cuda')
#     print("before clean:")
#     view_gpu()
    
#     torch.cuda.empty_cache()
    
#     print("after clean:")
#     view_gpu()



    job_id = now()
    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg) # task表示可以看作“软件包”，这之后就进入“软件包”minigpt4/tasks/__init__.py执行“软件包的初始化操作”
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    
    print("In train.py")
    view_model_device_allocation(model,cpu_only=True)
    check_frozen_parts(model,active_only=True)

    print(cfg.run_cfg.wandb_log)
    if cfg.run_cfg.wandb_log:
        wandb.login()
        wandb.init(project="minigptv", name=cfg.run_cfg.job_name)
        wandb.watch(model)

    logging.info("before clean, free memory: {}".format(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)))
    print("before clean:")
    view_gpu()
    
    torch.cuda.empty_cache()
    
    print("after clean:")
    view_gpu()
    print('CUDA usage:',torch.ones(1).cuda())

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
