import os
import torch
import random
import json
import csv
import numpy as np

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pathlib

from datasets import load_dataset
from torch.utils.data import Dataset


def load_data(data_name):
    if data_name == "openwebtext":
        data = load_dataset("openwebtext")
        return data

    elif data_name == "openwebtext10k":
        data_all = load_dataset("stas/openwebtext-10k")
        data = [p["text"] for p in data_all["train"]]
        return data

    elif data_name == "wikitext-103":
        data_all = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        data = data_all["text"]
        return data

    elif data_name == "bold":
        data = load_dataset("AlexaAI/bold")
        data = [p["prompts"] for p in data["train"] if p["domain"] == "profession"]
        data = sum(data, [])
        return data

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        data_name
    )
    out_dir = os.path.join(
        pathlib.Path("./data/scifact/").parent.absolute(), "datasets"
    )
    data_path = util.download_and_unzip(url, out_dir)
    if data_name == "msmarco":
        data_split = "dev"
    else:
        data_split = "test"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split=data_split
    )
    data = {}
    for qid, query in queries.items():
        data["q_{}".format(qid)] = query

    for pid, passage in corpus.items():
        if passage["title"] == "":
            data["p_{}".format(pid)] = passage["text"]
        else:
            data["p_{}".format(pid)] = passage["title"] + ": " + passage["text"]
    return data


class LmSeqsDataset(Dataset):
    """Custom Dataset wrapping language modeling sequences.
    Each sample will be retrieved by indexing the list of token_ids and their corresponding lengths.
    Input:
    ------
        params: `NameSpace` parameters
        data: `List[np.array[int]]
    """

    def __init__(self, params, data):
        self.params = params

        self.token_ids = np.array(data)
        self.lengths = np.array([len(t) for t in data])

        self.check()
        self.remove_empty_sequences()
        self.remove_long_sequences()
        self.remove_unknown_sequences()
        self.check()

    def __getitem__(self, index):
        return {"tokens": self.token_ids[index]}

    def __len__(self):
        return len(self.lengths)

    def check(self):
        """
        Some sanity checks
        """
        assert len(self.token_ids) == len(self.lengths)
        assert all(self.lengths[i] == len(self.token_ids[i]) for i in range(len(self.lengths)))

    def remove_long_sequences(self):
        """
        Sequences that are too long are split by chunk of max_model_input_size.
        """
        max_len = self.params["max_model_input_size"]
        indices = self.lengths > max_len

        def divide_chunks(l, n):
            return [l[i : i + n] for i in range(0, len(l), n)]

        new_tok_ids = []
        new_lengths = []
        if self.params["mlm"]:
            cls_id, sep_id = self.params["cls_token"], self.params["sep_token"]
        else:
            cls_id, sep_id = self.params["bos_token"], self.params["eos_token"]
        
        if not isinstance(cls_id, torch.LongTensor):
            cls_id = torch.LongTensor([cls_id])
        
        if not isinstance(sep_id, torch.LongTensor):
            sep_id = torch.LongTensor([sep_id])

        for seq_, len_ in zip(self.token_ids, self.lengths):
            # if seq_[0] != cls_id:
            #     seq_ = torch.cat((cls_id, seq_))
            #     len_ += 1
            # if seq_[-1] != sep_id:
            #     seq_ = torch.cat((seq_, sep_id))
            #     len_ += 1

            if len_ <= max_len:
                new_tok_ids.append(seq_)
                new_lengths.append(len_)
            else:
                sub_seqs = []
                for sub_s in divide_chunks(seq_, max_len - 2):
                    # if sub_s[0] != cls_id:
                    #     sub_s = torch.cat((cls_id, sub_s))
                    # if sub_s[-1] != sep_id:
                    #     sub_s = torch.cat((sub_s, sep_id))
                    # assert len(sub_s) <= max_len
                    # assert (sub_s[0] == cls_id) and (sub_s[-1] == sep_id), sub_s
                    sub_seqs.append(sub_s)

                new_tok_ids.extend(sub_seqs)
                new_lengths.extend([len(l) for l in sub_seqs])

        self.token_ids = np.array(new_tok_ids)
        self.lengths = np.array(new_lengths)

    def remove_empty_sequences(self):
        """
        Too short sequences are simply removed. This could be tuned.
        """
        indices = self.lengths > 11
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]

    def remove_unknown_sequences(self):
        """
        Remove sequences with a (too) high level of unknown tokens.
        """
        unk_token_id = self.params["unk_token"]
        init_size = len(self)
        unk_occs = np.array([np.count_nonzero(a == unk_token_id) for a in self.token_ids])
        indices = (unk_occs / self.lengths) < 0.5
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)

    def batch_sequences(self, batch):
        """
        Do the padding and transform into torch.tensor.
        """
        token_ids = [t["tokens"] for t in batch]
        lengths = [len(t) for t in token_ids]
        assert len(token_ids) == len(lengths)

        # Max for paddings
        max_seq_len_ = max(lengths)

        # Pad token ids
        if self.params["mlm"]:
            pad_idx = self.params["pad_token"]
        else:
            pad_idx = self.params["unk_token"]
        tk_ = [torch.cat((t, torch.tensor([pad_idx] * (max_seq_len_ - len(t))))) for t in token_ids]
        returnmasks = [torch.tensor([1] * len(t) + [0] * (max_seq_len_ - len(t))) for t in token_ids]
        assert len(tk_) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)

        tk_t = torch.stack(tk_, dim=0).long()
        returnmasks = torch.stack(returnmasks, dim=0).bool()

        _input, _output = dict(), dict()
        _input["input_ids"] = tk_t[:, :-1].contiguous()
        _input["attention_mask"] = returnmasks[:, :-1].contiguous()
        _output["input_ids"] = tk_t[:, 1:].contiguous()
        _output["attention_mask"] = returnmasks[:, 1:].contiguous()
        return {
            "input_tokens":_input,
            "output_tokens": _output,
        }


def randomcrop(x, ratio_min, ratio_max):
    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = x[start:end].clone()
    return crop


def build_mask(tensors):
    shapes = [x.shape for x in tensors]
    maxlength = max([len(x) for x in tensors])
    returnmasks = []
    ids = []
    for k, x in enumerate(tensors):
        returnmasks.append(torch.tensor([1] * len(x) + [0] * (maxlength - len(x))))
        ids.append(torch.cat((x, torch.tensor([0] * (maxlength - len(x))))))
    ids = torch.stack(ids, dim=0).long()
    returnmasks = torch.stack(returnmasks, dim=0).bool()
    return ids, returnmasks


def add_token(x, token):
    x = torch.cat((torch.tensor([token]), x))
    return x


def deleteword(x, p=0.1):
    mask = np.random.rand(len(x))
    x = [e for e, m in zip(x, mask) if m > p]
    return x


def replaceword(x, min_random, max_random, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else random.randint(min_random, max_random) for e, m in zip(x, mask)]
    return x


def maskword(x, mask_id, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else mask_id for e, m in zip(x, mask)]
    return x

def shuffleword(x, p=0.1):
    count = (np.random.rand(len(x)) < p).sum()
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(x)), k=count)
    to_shuffle = [x[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        x[old_index] = value
    return x


def apply_augmentation(x, opt):
    if opt.augmentation == "mask":
        return torch.tensor(maskword(x, mask_id=opt.mask_id, p=opt.prob_augmentation))
    elif opt.augmentation == "replace":
        return torch.tensor(
            replaceword(x, min_random=opt.start_id, max_random=opt.vocab_size - 1, p=opt.prob_augmentation)
        )
    elif opt.augmentation == "delete":
        return torch.tensor(deleteword(x, p=opt.prob_augmentation))
    elif opt.augmentation == "shuffle":
        return torch.tensor(shuffleword(x, p=opt.prob_augmentation))
    else:
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        return x


def add_bos_eos(x, bos_token_id, eos_token_id):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if bos_token_id is None and eos_token_id is not None:
        x = torch.cat([x.clone().detach(), torch.tensor([eos_token_id])])
    elif bos_token_id is not None and eos_token_id is None:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach()])
    elif bos_token_id is None and eos_token_id is None:
        pass
    else:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach(), torch.tensor([eos_token_id])])
    return x

