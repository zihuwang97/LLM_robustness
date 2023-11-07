#!/usr/bin/env python3

import os
import time
import sys
import torch
import logging
import json
import numpy as np
import random
import pickle

import torch.distributed as dist

from src.options import Options
from src import slurm, dist_utils, utils
from src import model_class


os.environ["TOKENIZER_PARALLELISM"] = "false"


if __name__ == "__main__":
    options = Options()
    opt = options.parse()

    torch.manual_seed(1)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    lm = model_class.lightning_module(opt)
    lm.fit()
