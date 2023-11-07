import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.base_lightning_module import base_lightning_module
from transformers import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration


class t5(nn.Module):
    """ GPT Language Model """

    def __init__(self, config):
        super().__init__()
        if config is None:
            config = {
                "model_name": "t5-small"
                }

        self.model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

      # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.model.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def forward(self, idx):
        logits = self.model(idx)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample=False):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample,
            )

class t5_module(base_lightning_module):
    def __init__(
        self,
        lr=2e-5,
        weight_decay=1e-4,
        do_validation=True,
        pretrained_checkpoint=None,
        warmup_steps=1000,
        training_steps=10000,
        model_config=None,
        opt=None,
        tokenizer=None,
        tokenizer_kw=None,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.do_validation = do_validation
        self.opt = opt

        self.model = t5(model_config)

        if pretrained_checkpoint is not None:
            state = torch.load(pretrained_checkpoint)["state_dict"]
            self.load_state_dict(state)

        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        else:
            self.tokenizer = tokenizer

        if tokenizer_kw is None:
            self.tokenizer_kw = dict(
                return_tensors="pt",
            )
        else:
            self.tokenizer_kw = tokenizer_kw

    def forward(self, idx):
        return self.model(idx)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens=20, do_sample=False):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample,
            )

    def training_and_validation_step(self, input_tokens, output_tokens, **kwargs):
        input_tokens, output_tokens = input_tokens["input_ids"], output_tokens["input_ids"]
        logits = self.model(input_tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output_tokens.view(-1), ignore_index=-1)
        return loss

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        self.model.model.transformer.wpe.weight.requires_grad = False

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                },
        }