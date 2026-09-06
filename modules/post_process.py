import re
import json
import logging
import torch
import numpy as np
from thefuzz import fuzz
from typing import Union, List
from pyhealth.tokenizer import Tokenizer
from pyhealth.models.utils import batch_to_multihot

from data.dataloader import load_mappings
from baselines.CoT.prompt import timespans

logger = logging.getLogger("pyhealth.trainer")
code2cond, code2proc, code2name = load_mappings()
conditions_names = list(code2cond.values())
procedures_names = list(code2proc.values())
drugs_names = list(code2name.values())


def parse_string(string):
    string = re.sub(r"\d\.", "", string)
    string = re.sub(r"[#*]", "", string)
    string = string.replace("\n\n", "\n")
    string = string.strip(": \n\t")
    return string

def match_concept_fuzz(input, pool):
    pool = list(pool)
    for drug in pool:
        if input == drug:
            return drug

    ratios = [fuzz.ratio(input, drug) for drug in pool]
    best_idx = np.argmax(ratios)
    if ratios[best_idx] >= 95:
        return pool[best_idx]

    for drug in pool:
        if input in drug:
            return drug

    return None

def match_drug_fuzz(input, pool=None):
    if pool is None: pool = drugs_names
    if input.lower() == "vitamin b12 and folic acid":
        return "vitamin b<n>12</n> and folic acid"
    elif input.lower() == 'vitamin b1, plain and in combination with vitamin b6 and b12':
        return "vitamin b<n>1</n>, plain and in combination with vitamin b<n>6</n> and b<n>12</n>"

    return match_concept_fuzz(input, pool)

def match_condition_fuzz(input, pool=None):
    if pool is None: pool = conditions_names
    return match_concept_fuzz(input, pool)

def match_procedure_fuzz(input, pool=None):
    if pool is None: pool = procedures_names
    return match_concept_fuzz(input, pool)


def extract_json_blocks(text):
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return json.loads(matches[0])


def post_process_binary(prob, response, output):
    conclusion = response.split("Prediction")[-1]
    prediction = re.findall(r"\d", conclusion)[0]
    return prob, (response, int(prediction)), output


def post_process_los(prob, response, output):
    prediction = response.split("Prediction")[-1].replace("days", "day").replace("weeks", "week")
    scores = [0] * len(timespans)
    for line in prediction.split("\n"):
        if ":" not in line: continue
        prefix, postfix = line.split(":")
        for timespan in timespans:
            if timespan in prefix and re.findall(r"\d", postfix):
                score = re.findall(r"\d", postfix)[0]
                scores[timespans.index(timespan)] = int(score)
                break

    assert sum(scores) > 0, f"Model output empty prediction for LOS. {response}"
    return prob, (response, scores), output


def post_process_drug(prob, response, output):
    prediction = response.split("Prediction")[-1].strip()

    scoring = {}
    for key, value in extract_json_blocks(prediction).items():
        assert isinstance(value, dict)
        assert 'reason' in value and 'score' in value
        value['score'] = int(value['score'])
        scoring[key] = dict(value)

    return prob, (response, scoring), output


POST_PROCESS_FN = {"mortality": post_process_binary, "readmission": post_process_binary,
                   "lenofstay": post_process_los, "drugrec": post_process_drug,
                   "diagnosis": post_process_drug}


def get_label_tokenizer(dataset, label_key, special_tokens=None) -> Tokenizer:
    label_tokenizer = Tokenizer(
        dataset.get_all_tokens(key=label_key),
        special_tokens=special_tokens,
    )
    return label_tokenizer


def prepare_labels(mode: str, labels: Union[List[str], List[List[str]]], label_tokenizer: Tokenizer):
    if mode in ["binary", 'mortality', 'readmission']:
        labels = label_tokenizer.convert_tokens_to_indices(labels)
        labels = torch.FloatTensor(labels).unsqueeze(-1)
    elif mode in ["multiclass", 'lenofstay']:
        labels = label_tokenizer.convert_tokens_to_indices(labels)
        labels = torch.LongTensor(labels)
    elif mode in ["multilabel", 'drugrec', "diagnosis"]:
        # convert to indices
        labels_index = label_tokenizer.batch_encode_2d(labels, padding=False, truncation=False)
        # convert to multihot
        num_labels = label_tokenizer.get_vocabulary_size()
        labels = batch_to_multihot(labels_index, num_labels)
    else:
        raise NotImplementedError
    return labels


# convert LLM output to logits
def get_logit(task, prediction, label_tokenizer):
    if task in ['mortality', 'readmission']:
        # prediction: int
        eps = 1e-15
        p = np.clip(int(prediction) / 10, eps, 1 - eps)
        return [np.log(p / (1 - p))]  # inverse sigmoid
    elif task in ['lenofstay']:
        # prediction: List[int]
        return [p / sum(prediction) for p in prediction]
    elif task in ['drugrec', 'diagnosis']:
        # prediction: List[str]
        logit = [0.] * label_tokenizer.get_vocabulary_size()
        if isinstance(prediction, list):
            for i, drug in enumerate(prediction):
                drug = match_drug_fuzz(drug, label_tokenizer.vocabulary.token2idx.keys())
                if drug:
                    pos = label_tokenizer.vocabulary(drug)
                    logit[pos] = 1 - 1e-4 * i
        elif isinstance(prediction, dict):
            for drug, item in prediction.items():
                drug = match_drug_fuzz(drug, label_tokenizer.vocabulary.token2idx.keys())
                if drug:
                    pos = label_tokenizer.vocabulary(drug)
                    logit[pos] = int(item['score']) / 5

        return logit
    else:
        raise ValueError()
