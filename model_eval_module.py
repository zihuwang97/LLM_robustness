import numpy as np
import torch

from datasets import load_dataset
import math
from collections import OrderedDict

import textattack
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder


# %%
class eval_base():
    def __init__(self):
        super().__init__()
    
    def load_data(self, task="nli-snli"):
        data_name = task[task.index("-")+1:]
        if data_name == "snli":
            # 0: entailment, 1: neutral, 2: contraction
            key_names = ["premise", "hypothesis"]
            split = "test"
        elif data_name == "multi_nli":
            # 0: entailment, 1: neutral, 2: contraction
            key_names = ["premise", "hypothesis"]
            split = "validation_matched"

        elif data_name == "boolq":
            # 0: false, 1: true
            key_names = ["question", "passage"]
            split = "validation"
        elif data_name == "cb":
            # 0: entailment, 1: contraction, 2: neutral
            key_names = ["premise", "hypothesis"]
            split = "validation"
        elif data_name == "copa":
            # 0: choice-1, 1: choice-2
            key_names = ["premise", "choice1", "choice2", "question"]
            split = "validation"
        elif data_name == "multirc":
            # 0: False, 1: True
            key_names = ["paragraph", "question", "answer"]
            split = "validation"
        elif data_name == "rte":
            # 0: entailment, 1: not entailment
            key_names = ["premise", "hypothesis"]
            split = "validation"
        elif data_name == "wic":
            # 0: False, 1: True
            key_names = ["word", "sentence1", "sentence2", "start1", "start2", "end1", "end2"]
            split = "validation"
        elif data_name == "wsc":
            # 0: False, 1: True
            key_names = ["text", "span1_index", "span2_index", "span1_text", "span2_text"]
            split = "validation"
        elif data_name == "wsc.fixed":
            # 0: False, 1: True
            key_names = ["text", "span1_index", "span2_index", "span1_text", "span2_text"]
            split = "validation"
        elif data_name == "axb":
            # 0: entailment, 1: not entailment
            key_names = ["sentence1", "sentence2"]
            split = "validation"
        elif data_name == "axg":
            # 0: entailment, 1: not entailment
            key_names = ["premise", "hypothesis"]
            split = "validation"

        dataset = load_dataset(data_name)
        labels = dataset[split]["label"]
        num = len(labels)
        dataset_dict = {name: dataset[split][name] for name in key_names}
        data_pairs = [[dataset_dict[name][ii] for name in key_names] for ii in range(num)]
        data_pairs_valid = [data_pairs[ii] for ii in range(num) if labels[ii] != -1]
        labels_valid = [labels[ii] for ii in range(num) if labels[ii] != -1]
        return data_pairs_valid, labels_valid


# %%

class eval_nli(eval_base):
    def __init__(self, model, tokenizer):
        super().__init__()
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
        # constraints = [RepeatModification(),
        #                 StopwordModification(),
        #                 WordEmbeddingDistance(min_cos_sim=0.9),
        #                 ]
        # transformation = textattack.transformations.WordSwapEmbedding(max_candidates=50)
        # search_method = textattack.search_methods.GreedyWordSwapWIR(wir_method="delete")

        # constraints = [RepeatModification(),
        #                 StopwordModification(),
        #                 UniversalSentenceEncoder(),
        #                 ]
        # transformation = textattack.transformations.WordInsertionMaskedLM(masked_language_model="roberta-base")
        # search_method = textattack.search_methods.GreedySearch()

        constraints = [UniversalSentenceEncoder(),]
        transformation = textattack.transformations.sentence_transformations.BackTranslation()
        search_method = textattack.search_methods.GreedySearch()

        # constraints = [RepeatModification(),
        #                 StopwordModification(),
        #                 UniversalSentenceEncoder(),
        #                 ]
        # transformation = textattack.transformations.WordInsertionRandomSynonym()
        # search_method = textattack.search_methods.GreedySearch()

        self.attack = textattack.Attack(goal_function, constraints, transformation, search_method)

    def generate_perturbed_dataset(self,
                                   text_batch,
                                   labels,
                                   ):
        text_batch_perturbed = []
        for (text_pair, ground_truth_label) in zip(text_batch, labels):
            example = OrderedDict({"premise": text_pair[0], "hypothesis": text_pair[1]})
            attack_result = self.attack.attack(example, ground_truth_label)

            if not "premise" in attack_result.perturbed_result.attacked_text._text_input:
                text_batch_perturbed.append(text_pair)
            elif not "hypothesis" in attack_result.perturbed_result.attacked_text._text_input:
                text_batch_perturbed.append(text_pair)
            else:
                example_perturbed = [attack_result.perturbed_result.attacked_text._text_input["premise"],
                                    attack_result.perturbed_result.attacked_text._text_input["hypothesis"],
                                    ]
                text_batch_perturbed.append(example_perturbed)
        return text_batch_perturbed

    def evaluation(self, 
                   model, 
                   tokenizer,
                   tokenizer_kw,
                   data_name="nli-snli", 
                   perturbation=None, 
                   batch_size=10,
                   ):
        data_pairs, labels = self.load_data(task=data_name)
        num_total = len(labels)
        num_batches = math.ceil(num_total / batch_size)
        num_correct = 0
        num_total = 0
        for ii in range(0, num_batches):
            print(ii, "/", num_batches)
            X = data_pairs[ii*batch_size:(ii+1)*batch_size]
            Y = labels[ii*batch_size:(ii+1)*batch_size]
            
            if perturbation is not None:
                X = self.generate_perturbed_dataset(X, Y)

            X_tokens = tokenizer(X, **tokenizer_kw)
            input_ids = X_tokens.input_ids.to(model.device)
            attention_mask = X_tokens.attention_mask.to(model.device)
            with torch.no_grad():
                logits = model(input_ids, attention_mask).logits
            predictions = logits.argmax(dim=1)
            num_correct += sum([predictions[jj].item() == Y[jj] for jj in range(len(Y))])
            num_total += batch_size
        accuracy = num_correct / num_total
        print(accuracy)


