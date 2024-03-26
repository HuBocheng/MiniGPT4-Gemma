"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

from omegaconf import OmegaConf

from minigpt4.common.registry import registry

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.tasks import *

# os.path.abspath(__file__) 获取当前文件（__init__.py）的绝对路径，
# 然后 os.path.dirname() 获取这个文件所在的目录的路径，也就是项目的根目录
root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)  # 注册项目根目录
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)  # 注册项目根目录的上一级目录
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)  # 注册缓存目录

registry.register("MAX_INT", sys.maxsize)  # 2^63 - 1
registry.register("SPLIT_NAMES", ["train", "val", "test"])  # 默认的分割名称，没用
