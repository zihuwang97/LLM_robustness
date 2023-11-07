#!/usr/bin/env python3
import torch
import os
import math
import time
import sys

import numpy as np
import argparse
import json

import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.GPT_model import gpt_module
from src.T5_model import t5_module
from src.auto_model import auto_module

from transformers import AutoTokenizer

from model_eval_module import eval_nli

from transformers import T5Tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--pretrained_checkpoint", type=str, default="logs/nli_snli_distilbert-base-uncased_0.0001/version_0/checkpoints/epoch=4-step=5360.ckpt")
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--task_name", type=str, default="nli-snli")
parser.add_argument("--max_length", type=int, default=512)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
def main(args):
    model_name = args.model_name
    batch_size = args.batch_size
    pretrained_checkpoint = args.pretrained_checkpoint
    task_name = args.task_name
    max_length = args.max_length

    _gpu_ids = [i for i in range(torch.cuda.device_count())]
    _gpu_ids = [_gpu_ids[0]]
    if torch.cuda.device_count() > 0:
        batch_size = batch_size * len(_gpu_ids)
    args.per_gpu_batch_size = batch_size
    args.per_gpu_eval_batch_size = batch_size
    args.num_workers = 5

    config_params = {
        "model_name": model_name,
        "batch_size": batch_size,
        "pretrained_checkpoint": pretrained_checkpoint,
        "opt": args,
    }

    if "gpt" in model_name:
        model = gpt_module(**config_params)

    elif "bert" in model_name:
        model = auto_module(**config_params)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_kw = dict(
            return_tensors="pt",
            padding=True,
            max_length=max_length,
        )

    elif "t5" in model_name:
        model = t5_module(**config_params)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    # --> delete
    else:
        model = auto_module(**config_params)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_kw = dict(
            return_tensors="pt",
            padding=True,
            # max_length=max_length,
        )


    model.eval()
    model = model.cuda(_gpu_ids[0]) if _gpu_ids != [] else model

    if "nli" in task_name:
        eval = eval_nli(model.model, tokenizer)
        eval.evaluation(model=model,
                        tokenizer=tokenizer,
                        tokenizer_kw=tokenizer_kw,
                        data_name=task_name,
                        perturbation='adversarial',
                        batch_size=batch_size,
                        )


if __name__ == "__main__":
    main(args)


