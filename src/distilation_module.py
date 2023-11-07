from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
# from transformers import AutoModelForSequenceClassification
from transformers import DistilBertModel
from transformers import AutoTokenizer, AutoModel

from datasets import load_dataset, load_from_disk

from src.base_lightning_module import base_lightning_module
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, Dataset

from src.text_transformation import TextTransformation

import pandas as pd
from datasets import Dataset, DatasetDict

from textattack.transformations import CompositeTransformation
from textattack.transformations import WordSwapRandomCharacterDeletion, WordSwapQWERTY, WordInsertionRandomSynonym
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.augmentation import Augmenter
from textattack.shared import utils

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

class SeqDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        return {"sentence": self.ds["sentence"][index],
                "sentence_perturbed": self.ds["sentence_perturbed"][index],
                "label": self.ds["label"][index],
                }

    def __len__(self):
        return len(self.ds["label"])


class Collator(object):
    def __init__(self, tokenizer, tokenizer_kw):
        self.tokenizer = tokenizer
        self.tokenizer_kw = tokenizer_kw

    def __call__(self, sample_text_label_batch):
        sample_batch, perturbed_batch, label_batch = [], [], []
        for data in sample_text_label_batch:
            # print(data)
            s, s_perturbed, label = data["sentence"], data["sentence_perturbed"], data["label"]
            sample_batch.append(s)
            perturbed_batch.append(s_perturbed)
            label_batch.append(label)

        encoded_text = self.tokenizer(sample_batch, **self.tokenizer_kw)
        encoded_perturbed_text = self.tokenizer(perturbed_batch, **self.tokenizer_kw)
        label_batch = torch.tensor(label_batch)
        output = {"input_ids": encoded_text.input_ids,
                  "attention_mask": encoded_text.attention_mask,
                  "perturbed_ids": encoded_perturbed_text.input_ids,
                  "perturbed_attention_mask": encoded_perturbed_text.attention_mask,
                  "label": label_batch,
                }
        return output
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class perturbation:
    def __init__(self):
        # FUTURE: consider other trans
        transformation = CompositeTransformation([WordSwapRandomCharacterDeletion(), 
                                                #   WordSwapQWERTY(),
                                                  WordInsertionRandomSynonym(),])
        # POSSIBLY: consider word embedding
        constraints = [RepeatModification(), StopwordModification()]
        self.augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.5, transformations_per_example=1)
    def __call__(self, sample):
        perturbed = self.augmenter.augment(sample)
        return perturbed

class compute_score:
    def __call__(self, target_model, target_tokenizer, u, v):
        encoded_u = target_tokenizer(u, padding=True, truncation=True, return_tensors='pt')
        encoded_v = target_tokenizer(v, padding=True, truncation=True, return_tensors='pt')

        u_input_ids = encoded_u.input_ids.to(target_model.device)
        u_attention_mask = encoded_u.attention_mask.to(target_model.device)
        v_input_ids = encoded_v.input_ids.to(target_model.device)
        v_attention_mask = encoded_v.attention_mask.to(target_model.device)
        # NOTE: mean or last word pooling?
        z_u = mean_pooling(target_model(u_input_ids, u_attention_mask), 
                           u_attention_mask)
        z_v = mean_pooling(target_model(v_input_ids, v_attention_mask), 
                           v_attention_mask)
        # NOTE: dot product or cosine similarity?
        z_u = F.normalize(z_u, dim=-1).squeeze()
        z_v = F.normalize(z_v, dim=-1).squeeze()
        sim = torch.sum(z_u @ z_v.T)
        return sim


class data_module_distilation(LightningDataModule):
    def __init__(
        self,
        data_name,
        tokenizer,
        tokenizer_kw,
        batch_size,
        num_workers,
        do_validation=True,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.do_validation = do_validation
        self.tokenizer = tokenizer
        self.tokenizer_kw = tokenizer_kw
        
        # data_train, _, data_valid, _ = self.load_data(data_name)

        # # -->> replace with real perturbations
        # # -->> maybe do not used universal sentence encoding as constraint
        # perturbation_module = perturbation()
        # score_compute = compute_score()
        # # target_model = tf.keras.Sequential([hub.KerasLayer("saved_model", trainable=True),])
        
        # target_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens')
        # target_model = AutoModel.from_pretrained('sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens').cuda()

        # sentence_train = [sample[0] for sample in data_train] + [sample[1] for sample in data_train]
        # sentence_valid = [sample[0] for sample in data_valid] + [sample[1] for sample in data_valid]

        # # --> reduce number of samples
        # sentence_train = sentence_train[500000:]
        # sentence_valid = sentence_valid[10000:]
        
        # df_train = pd.DataFrame.from_dict({"sentence": sentence_train})
        # tds = Dataset.from_pandas(df_train)

        # df_valid = pd.DataFrame.from_dict({"sentence": sentence_valid})
        # vds = Dataset.from_pandas(df_valid)
    
        # ds = DatasetDict()
        # ds["train"] = tds
        # ds["valid"] = vds
        
        # def create_perturbation(sample):
        #     sample_perturbed = perturbation_module(sample["sentence"])
        #     sample["sentence_perturbed"] = sample_perturbed[0]
        #     return sample

        # ds["train"] = ds["train"].map(lambda x: create_perturbation(x))
        # ds["valid"] = ds["valid"].map(lambda x: create_perturbation(x))
       
        # def create_label(sample):
        #     label = score_compute(target_model, target_tokenizer, 
        #                           [sample["sentence"]], [sample["sentence_perturbed"]])
        #     sample["label"] = label.item()
        #     return sample

        # ds["train"] = ds["train"].map(lambda x: create_label(x))
        # ds["valid"] = ds["valid"].map(lambda x: create_label(x))
        # ds.save_to_disk('attacked_snli_2')
        ds = load_from_disk('attacked_snli')
        
        self.SeqDataset_train = SeqDataset(ds["train"])
        self.SeqDataset_valid = SeqDataset(ds["valid"])
        self.Collate_train = Collator(tokenizer, tokenizer_kw)
        self.Collate_valid = Collator(tokenizer, tokenizer_kw)

    def train_dataloader(self):
        # sampler = RandomSampler(self.SeqDataset_train)
        train_dataloader = DataLoader(
            self.SeqDataset_train,
            # sampler=sampler,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.Collate_train,
            pin_memory=True
        )
        return train_dataloader

    def val_dataloader(self):
        valid_dataloader = DataLoader(
            self.SeqDataset_valid,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.Collate_valid,
            pin_memory=True
        )
        return valid_dataloader

    # def load_data(self, data_name):
    #     if data_name == "snli":
    #         key_names = ["premise", "hypothesis"]
    #     elif data_name == "multi-nli":
    #         key_names = ["premise", "hypothesis"]
    #     elif data_name == "boolq":
    #         key_names = ["question", "passage"]
    #     elif data_name == "cb":
    #         key_names = ["question", "passage"]
    #     elif data_name == "copa":
    #         key_names = ["premise", "choice1", "choice2"]
    #     elif data_name == "multirc":
    #         key_names = ["paragraph", "question", "answer"]
    #     elif data_name == "rte":
    #         key_names = ["premise", "hypothesis"]
    #     elif data_name == "wic":
    #         key_names = [""]
    #     elif data_name == "wsc":
    #         key_names = [""]
    #     elif data_name == "wsc.fixed":
    #         key_names = [""]
    #     elif data_name == "axb":
    #         key_names = [""]
    #     elif data_name == "axg":
    #         key_names = ["premise", "hypothesis"]

    #     dataset = load_dataset(data_name)
    #     labels_train, labels_valid = dataset["train"]["label"], dataset["validation"]["label"]
    #     dataset_dict_train = {name: dataset["train"][name] for name in key_names}
    #     dataset_dict_valid = {name: dataset["validation"][name] for name in key_names}

    #     data_pairs_train = [[dataset_dict_train[name][ii] for name in key_names] for ii in range(len(labels_train))]
    #     data_pairs_valid = [[dataset_dict_valid[name][ii] for name in key_names] for ii in range(len(labels_valid))]

    #     num_train, num_valid = len(data_pairs_train), len(data_pairs_valid)

    #     data_pairs_train = [data_pairs_train[ii] for ii in range(num_train) if labels_train[ii] != -1]
    #     data_pairs_valid = [data_pairs_valid[ii] for ii in range(num_valid) if labels_valid[ii] != -1]
    #     labels_train = [labels_train[ii] for ii in range(num_train) if labels_train[ii] != -1]
    #     labels_valid = [labels_valid[ii] for ii in range(num_valid) if labels_valid[ii] != -1]
    #     return data_pairs_train, labels_train, data_pairs_valid, labels_valid
    
    # def data_augmentation(self, dataset, labels, transformation, perturbations):
    #     dataset_aug, labels_aug = dataset, labels
    #     for sample_text, label in zip(dataset, labels):
    #         if not isinstance(sample_text, list):
    #             sample_text = list(sample_text)
    #         for perturbation in perturbations:
    #             sample_text_perturbed = []
    #             for text in sample_text:
    #                 sample_text_perturbed.append(transformation.apply(text, perturbation))

    #             if not isinstance(sample_text, list):
    #                 sample_text_perturbed = sample_text_perturbed[0]
    #             dataset_aug.append(sample_text_perturbed)
    #             labels_aug.append(label)
    #     return dataset_aug, labels_aug


class distilation_module(base_lightning_module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        lr=2e-5,
        weight_decay=1e-4,
        do_validation=True,
        pretrained_checkpoint=None,
        warmup_steps=1000,
        total_steps=10000,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.do_validation = do_validation

        self.model = DistilBertModel.from_pretrained(model_name)

        if pretrained_checkpoint is not None:
            state = torch.load(pretrained_checkpoint)["state_dict"]
            self.load_state_dict(state)

        self.warmup_steps = warmup_steps
        self.training_steps = total_steps
        self.model_name = model_name
        self.criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_and_validation_step(self, input_ids, attention_mask, perturbed_ids, perturbed_attention_mask, label, **kwargs):
        input_ids, attention_mask = input_ids.to(self.model.device), attention_mask.to(self.model.device)
        perturbed_ids, perturbed_attention_mask = perturbed_ids.to(self.model.device), perturbed_attention_mask.to(self.model.device)
        
        # NOTE: output pooling
        z_u = mean_pooling(self.model(input_ids, attention_mask), attention_mask)
        z_v = mean_pooling(self.model(perturbed_ids, perturbed_attention_mask), perturbed_attention_mask)

        # NOTE: cosine or dot?
        z_u = F.normalize(z_u, dim=-1)
        z_v = F.normalize(z_v, dim=-1)
        pred = torch.sum(z_u * z_v, dim=-1)

        loss = self.criterion(pred, label)
        return loss

    def configure_optimizers(self):
        param_optimizer = list(self.parameters())
        optimizer = torch.optim.AdamW(
            param_optimizer, lr=self.lr
        )
        return {
            "optimizer": optimizer,
        }
        # param_optimizer = list(self.named_parameters())
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [
        #             p for n, p in param_optimizer
        #         ],
        #     }
        # ]
        # optimizer = torch.optim.AdamW(
        #     optimizer_grouped_parameters, lr=self.lr
        # )
        # lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.training_steps,
        # )
        # return {
        #     "optimizer": optimizer,
        #     # "lr_scheduler": {
        #     #     "scheduler": lr_scheduler,
        #     #     "monitor": "val_epoch_loss",
        #     # },
        # }


