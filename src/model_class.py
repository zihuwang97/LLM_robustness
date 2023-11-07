import tqdm
import math
import torch

from transformers import AutoTokenizer
from src.auto_model import data_module_auto_model, auto_module

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer





class lightning_module:
    def __init__(
        self,
        opt,
        do_validation=True,
        strategy=None,
        **kw,
    ):
        if strategy is None:
            strategy = "dp" if torch.cuda.device_count() > 1 else None

        self.task_name, self.data_name = opt.task_name.split("-")
        self.model_name = opt.model_name
        self.num_labels = opt.num_labels
        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.batch_size = opt.per_gpu_batch_size
        self.num_workers = opt.num_workers
        self.pooling = opt.pooling
        self.max_epochs = opt.max_epochs
        self.max_length = opt.max_length
        self.data_augmentation_methods = opt.data_augmentation
        self.pretrained_checkpoint = opt.pretrained_checkpoint
        self.do_validation = do_validation
        self.max_steps = -1
        self.strategy = 'ddp'#strategy

    def _get_data_module(self,
                          task_name,
                          data_name,
                          tokenizer, 
                          tokenizer_kw,
                          batch_size,
                          num_workers,
                          do_validation=True,
                          data_augmentation_methods=None,
                          ):
        if task_name.startswith("nli"):
            return data_module_auto_model(
                data_name,
                tokenizer,
                tokenizer_kw,
                batch_size,
                num_workers,
                do_validation=do_validation,
                perturbations=data_augmentation_methods,
                )

    def fit(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer_kw = dict(              
            return_tensors="pt",        
            padding=True,
            max_length=self.max_length,
        )
        dm = self._get_data_module(self.task_name,
                                   self.data_name,
                                   tokenizer,
                                   tokenizer_kw,
                                   self.batch_size,
                                   self.num_workers,
                                   self.do_validation,
                                   self.data_augmentation_methods,
                                   )
        num_batch_total = len(dm.train_dataloader())
        accumulate_grad_batches = 1
        total_steps = math.ceil(num_batch_total * self.max_epochs / accumulate_grad_batches)
        warmup_steps = math.ceil(total_steps * 0.05 / accumulate_grad_batches)

        config_params = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "do_validation": self.do_validation,
            "pretrained_checkpoint": self.pretrained_checkpoint,
            "pooling": self.pooling,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        }
        if self.task_name == "nli":
            model = auto_module(
                **config_params
            )

        # dm.use_adv_collate(model.model, tokenizer)

        if "/" in self.model_name:
            model_name = model_name.replace("/", "_")
        exp_name = "{}_{}_{}_{}".format(self.task_name,
                                        self.data_name,
                                        self.model_name,
                                        self.lr,
        )
        tb_logger = pl_loggers.TensorBoardLogger("./logs", name=exp_name)
        trainer = Trainer(
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            # gpus=torch.cuda.device_count(),
            devices=1,              # num of gpus
            strategy=self.strategy,
            log_every_n_steps=1,
            callbacks=[LearningRateMonitor()],
            logger=[tb_logger],
            precision=16,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=1.,
        )

        trainer.fit(model, datamodule=dm)


