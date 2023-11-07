import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification

from datasets import load_dataset

from src.base_lightning_module import base_lightning_module
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, Dataset

import textattack
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from collections import OrderedDict


class SeqDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        # self.input_ids = dataset["input_ids"]
        # self.attention_mask = dataset["attention_mask"]
        self.labels = labels

    def __getitem__(self, index):
        return {"sample_text": self.dataset[index],
                "label": self.labels[index],
                }
        # return {"input_ids": self.input_ids[index],
        #         "attention_mask": self.attention_mask[index],
        #         "label": self.labels[index],
        #         }

    def __len__(self):
        return len(self.labels)


class Collator(object):
    def __init__(self, tokenizer, tokenizer_kw):
        self.tokenizer = tokenizer
        self.tokenizer_kw = tokenizer_kw

    def __call__(self, sample_text_label_batch):
        sample_text_batch, label_batch = [], []
        for data in sample_text_label_batch:
            sample_text, label = data["sample_text"], data["label"]
            sample_text_batch.append(sample_text)
            label_batch.append(label)
        encoded_text = self.tokenizer(sample_text_batch, **self.tokenizer_kw)
        label_batch = torch.stack(label_batch)
        output = {"input_ids": encoded_text.input_ids,
                "attention_mask": encoded_text.attention_mask,
                "label": label_batch,
                }
        return output


class Collator_adv(object):
    def __init__(self, model, tokenizer):
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
        constraints = [RepeatModification(),
                        StopwordModification(),
                        WordEmbeddingDistance(min_cos_sim=0.9),
                        ]
        transformation = textattack.transformations.WordSwapEmbedding(max_candidates=50)
        search_method = textattack.search_methods.GreedyWordSwapWIR(wir_method="delete")
        self.attack = textattack.Attack(goal_function, constraints, transformation, search_method)
        self.tokenizer = tokenizer

    def __call__(self, sample_text_label_batch):
        sample_text_batch, label_batch = [], []
        for data in sample_text_label_batch:
            sample_text, label = data["sample_text"], data["label"]

            example = OrderedDict({"premise": sample_text[0], "hypothesis": sample_text[1]})
            attack_result = self.attack.attack(example, label)

            if not "premise" in attack_result.perturbed_result.attacked_text._text_input:
                sample_text_batch.append(sample_text)
            elif not "hypothesis" in attack_result.perturbed_result.attacked_text._text_input:
                sample_text_batch.append(sample_text)
            else:
                example_perturbed = [attack_result.perturbed_result.attacked_text._text_input["premise"],
                                    attack_result.perturbed_result.attacked_text._text_input["hypothesis"],
                                    ]
                sample_text_batch.append(example_perturbed)
            label_batch.append(label)
        encoded_text = self.tokenizer(sample_text_batch, **self.tokenizer_kw)
        label_batch = torch.stack(label_batch)
        output = {"input_ids": encoded_text.input_ids,
                "attention_mask": encoded_text.attention_mask,
                "label": label_batch,
                }
        return output


class data_module_auto_model(LightningDataModule):
    def __init__(
        self,
        data_name,
        tokenizer,
        tokenizer_kw,
        batch_size,
        num_workers,
        do_validation=True,
        perturbations=None,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.do_validation = do_validation
        self.tokenizer = tokenizer
        self.tokenizer_kw = tokenizer_kw
        self.Collate_train = Collator(tokenizer, tokenizer_kw)
        self.Collate_valid = Collator(tokenizer, tokenizer_kw)
        data_train, labels_train, data_valid, labels_valid = self.load_data(data_name)

        ## --> delete
        # data_train = data_train[0:1000]
        # labels_train = labels_train[0:1000]
        # data_valid = data_valid[0:1000]
        # labels_valid = labels_valid[0:1000]

        labels_train = torch.LongTensor(labels_train)
        labels_valid = torch.LongTensor(labels_valid)
        self.SeqDataset_train = SeqDataset(data_train, labels_train)
        self.SeqDataset_valid = SeqDataset(data_valid, labels_valid)

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.SeqDataset_train,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.Collate_train,
            pin_memory=True,
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
            pin_memory=True,
        )
        return valid_dataloader

    def load_data(self, data_name):
        if data_name == "snli":
            key_names = ["premise", "hypothesis"]
        elif data_name == "multi-nli":
            key_names = ["premise", "hypothesis"]
        elif data_name == "boolq":
            key_names = ["question", "passage"]
        elif data_name == "cb":
            key_names = ["question", "passage"]
        elif data_name == "copa":
            key_names = ["premise", "choice1", "choice2"]
        elif data_name == "multirc":
            key_names = ["paragraph", "question", "answer"]
        elif data_name == "rte":
            key_names = ["premise", "hypothesis"]
        elif data_name == "wic":
            key_names = [""]
        elif data_name == "wsc":
            key_names = [""]
        elif data_name == "wsc.fixed":
            key_names = [""]
        elif data_name == "axb":
            key_names = [""]
        elif data_name == "axg":
            key_names = ["premise", "hypothesis"]

        dataset = load_dataset(data_name)
        labels_train, labels_valid = dataset["train"]["label"], dataset["validation"]["label"]
        dataset_dict_train = {name: dataset["train"][name] for name in key_names}
        dataset_dict_valid = {name: dataset["validation"][name] for name in key_names}

        data_pairs_train = [[dataset_dict_train[name][ii] for name in key_names] for ii in range(len(labels_train))]
        data_pairs_valid = [[dataset_dict_valid[name][ii] for name in key_names] for ii in range(len(labels_valid))]

        num_train, num_valid = len(data_pairs_train), len(data_pairs_valid)

        data_pairs_train = [data_pairs_train[ii] for ii in range(num_train) if labels_train[ii] != -1]
        data_pairs_valid = [data_pairs_valid[ii] for ii in range(num_valid) if labels_valid[ii] != -1]
        labels_train = [labels_train[ii] for ii in range(num_train) if labels_train[ii] != -1]
        labels_valid = [labels_valid[ii] for ii in range(num_valid) if labels_valid[ii] != -1]
        return data_pairs_train, labels_train, data_pairs_valid, labels_valid
    
    def data_augmentation(self, dataset, labels, transformation, perturbations):
        dataset_aug, labels_aug = dataset, labels
        for sample_text, label in zip(dataset, labels):
            if not isinstance(sample_text, list):
                sample_text = list(sample_text)
            for perturbation in perturbations:
                sample_text_perturbed = []
                for text in sample_text:
                    sample_text_perturbed.append(transformation.apply(text, perturbation))

                if not isinstance(sample_text, list):
                    sample_text_perturbed = sample_text_perturbed[0]
                dataset_aug.append(sample_text_perturbed)
                labels_aug.append(label)
        return dataset_aug, labels_aug
    

# class DistilBertForSequenceClassification(nn.Module):
#     def __init__(self, model_name, num_labels=3):
#         super().__init__()
#         model_huggingface = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
#         self.distilbert = model_huggingface.distilbert
#         self.pre_classifier = model_huggingface.pre_classifier
#         self.classifier = model_huggingface.classifier
#         self.dropout = model_huggingface.dropout
#         self.num_labels = num_labels

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         return_embeddings = False,
#         request_embedding_gradient = False,
#     ):

#         output_attentions = output_attentions if output_attentions is not None else self.distilbert.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.distilbert.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.distilbert.config.use_return_dict

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         device = input_ids.device if input_ids is not None else inputs_embeds.device

#         if attention_mask is None:
#             attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

#         head_mask = self.distilbert.get_head_mask(head_mask, self.distilbert.config.num_hidden_layers)

#         # generation embeddings
#         if input_ids is not None:
#             input_embeds = self.distilbert.embeddings.word_embeddings(input_ids)
        
#         seq_length = input_embeds.size(1)

#         if hasattr(self.distilbert.embeddings, "position_ids"):
#             position_ids = self.distilbert.embeddings.position_ids[:, :seq_length]
#         else:
#             position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
#             position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

#         position_embeddings = self.distilbert.embeddings.position_embeddings(position_ids)

#         embeddings = input_embeds + position_embeddings
#         embeddings = self.distilbert.embeddings.LayerNorm(embeddings)
#         embeddings = self.distilbert.embeddings.dropout(embeddings)

#         if return_embeddings == True:
#             return embeddings

#         if request_embedding_gradient == True:
#             self.hidden_states = embeddings
#             self.hidden_states.requires_grad_()
#             self.hidden_states.retain_grad()
#             embeddings = self.hidden_states

#         # generate distilbert_output
#         all_hidden_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None

#         hidden_state = embeddings
#         for i, layer_module in enumerate(self.distilbert.transformer.layer):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_state,)

#             layer_outputs = layer_module(
#                 x=hidden_state, attn_mask=attention_mask, head_mask=head_mask[i], output_attentions=output_attentions
#             )
#             hidden_state = layer_outputs[-1]

#             if output_attentions:
#                 if len(layer_outputs) != 2:
#                     raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")

#                 attentions = layer_outputs[0]
#                 all_attentions = all_attentions + (attentions,)
#             else:
#                 if len(layer_outputs) != 1:
#                     raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_state,)

#         distilbert_output = tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)

#         hidden_state = distilbert_output[0]
#         pooled_output = hidden_state[:, 0] 
#         pooled_output = self.pre_classifier(pooled_output)
#         pooled_output = nn.ReLU()(pooled_output)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         return logits

#     @property
#     def device(self):
#         return self.distilbert.device


class auto_module(base_lightning_module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_labels=3,
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

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        if pretrained_checkpoint is not None:
            state = torch.load(pretrained_checkpoint)["state_dict"]
            self.load_state_dict(state)

        self.warmup_steps = warmup_steps
        self.training_steps = total_steps
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, 
                input_ids, 
                attention_mask=None,
                ):
        
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          )

    def training_and_validation_step(self, input_ids, attention_mask, label, **kwargs):
        input_ids, attention_mask = input_ids.to(self.model.device), attention_mask.to(self.model.device)
        outputs = self.model(input_ids, attention_mask).logits
        loss = self.criterion(outputs, label)
        return loss

    def configure_optimizers(self):
        param_optimizer = list(self.parameters())
        optimizer = torch.optim.AdamW(
            param_optimizer, lr=self.lr
        )
        return {
            "optimizer": optimizer,
        }
