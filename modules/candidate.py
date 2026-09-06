import torch
import numpy as np
from argparse import Namespace
from collections import defaultdict

from utils.load_model import get_model
from data.dataloader import load_mappings
from modules.kg import graph, get_neighbors
from modules.blacklists import edge_rule_filtering
from modules.motif import is_D, is_C

_, _, code2drug = load_mappings()


def load_mortality_candidate_model(dataset, task, full_set, model_name = "AdaCare"):
    config = Namespace(dataset=dataset, task=task, embedding_dim=128, method=model_name, exp='base')
    model = get_model(config, mode='binary', feature_keys=["conditions", "procedures", 'drugs'], full_dataset=full_set)
    ckpt_path = f"./saves/{dataset}-{task}/{model_name}/base/run0/best.ckpt"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_readmission_candidate_model(dataset, task, full_set, model_name='StageNet'):
    config = Namespace(dataset=dataset, task=task, embedding_dim=128, method=model_name, exp='base')
    model = get_model(config, mode='binary', feature_keys=["conditions", "procedures", 'drugs'], full_dataset=full_set)
    ckpt_path = f"./saves/{dataset}-{task}/{model_name}/base/run0/best.ckpt"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_lenofstay_candidate_model(dataset, task, full_set, model_name='StageNet'):
    config = Namespace(dataset=dataset, task=task, embedding_dim=128, method=model_name, exp='base')
    model = get_model(config, mode='multiclass', feature_keys=["conditions", "procedures"], full_dataset=full_set)
    ckpt_path = f"./saves/{dataset}-{task}/{model_name}/base/run0/best.ckpt"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_drugrec_candidate_model(dataset, task, full_set, model_name='StageNet'):
    config = Namespace(dataset=dataset, task=task, embedding_dim=128, method=model_name, exp='base')
    model = get_model(config, mode='multilabel', feature_keys=["conditions", "procedures"], full_dataset=full_set)
    ckpt_path = f"./saves/{dataset}-{task}/{model_name}/base/run0/best.ckpt"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_diagnosis_candidate_model(dataset, task, full_set, model_name='StageNet'):
    config = Namespace(dataset=dataset, task=task, embedding_dim=128, method=model_name, exp='base')
    model = get_model(config, mode='multilabel', feature_keys=["conditions", "procedures", 'drugs'], full_dataset=full_set)
    ckpt_path = f"./saves/{dataset}-{task}/{model_name}/base/run0/best.ckpt"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


def padding(concept_visits):
    for visit in concept_visits:
        if len(visit) == 0:
            visit.append('<unk>')
    return concept_visits


def get_candidate_mort(fn, model, concepts, topk=1000):
    conditions, procedures, drugs = concepts
    if fn == 'dl':
        conditions, procedures, drugs = padding(conditions), padding(procedures), padding(drugs)
        kwargs = {'conditions': [conditions], 'procedures': [procedures], "drugs": [drugs], 'label': [0]}
        with torch.no_grad():
            output = model(**kwargs)

        predicted_mort = output['y_prob'][0].item()

    elif fn == 'kg':
        target_nodes = [e for e in graph.entities if 'mortality' in e.lower()]
        target_node_ids = [graph.entity2id[e] for e in target_nodes]

        candidate_morts = defaultdict(int)
        for c in conditions + procedures + drugs:
            triplets = get_neighbors(c, target_node_ids, direction='forward')
            for c, _, mort in triplets:
                candidate_morts[mort] += 1

            triplets = get_neighbors(c, target_node_ids, direction='backward')
            for mort, _, c in triplets:
                candidate_morts[mort] += 1

        candidate_morts = sorted(candidate_morts.items(), key=lambda x: x[1], reverse=True)
        predicted_mort = [item[0] for item in candidate_morts][:topk]

    else:
        raise ValueError()

    return predicted_mort


def get_candidate_read(fn, model, concepts, topk=1000):
    conditions, procedures, drugs = concepts
    if fn == 'kg':
        target_nodes = [e for e in graph.entities if 'readmission' in e.lower() or 'readmit' in e.lower()]
        target_node_ids = [graph.entity2id[e] for e in target_nodes]

        predicted_read = defaultdict(int)
        for c in conditions + procedures + drugs:
            triplets = get_neighbors(c, target_node_ids, direction='forward')
            for c, _, read in triplets:
                predicted_read[read] += 1

            triplets = get_neighbors(c, target_node_ids, direction='backward')
            for read, _, c in triplets:
                predicted_read[read] += 1

        predicted_read = sorted(predicted_read.items(), key=lambda x: x[1], reverse=True)
        predicted_read = [item[0] for item in predicted_read][:topk]

    else:
        predicted_read = get_candidate_mort(fn, model, concepts, topk)

    return predicted_read


def get_candidate_los(fn, model, concepts):
    conditions, procedures, _ = concepts
    if fn == 'dl':
        conditions, procedures = padding(conditions), padding(procedures)
        kwargs = {'conditions': [conditions], 'procedures': [procedures], 'label': [0]}
        with torch.no_grad():
            output = model(**kwargs)

        solution_score = output['y_prob'][0].cpu().numpy()  # (9,)
        predicted_los = [int(round(x, 1) * 10) for x in solution_score]
    else:
        predicted_los = [0] * 9

    return predicted_los


def get_candidate_drugs(fn, model, label_tokenizer, concepts, topk=1000):
    conditions, procedures, drugs = concepts
    if fn == 'dl':
        conditions, procedures = padding(conditions), padding(procedures)
        kwargs = {'conditions': [conditions], 'procedures': [procedures], 'drugs': [drugs]}
        with torch.no_grad():
            output = model(**kwargs)

        solution_score = output['y_prob'][0].cpu().numpy()  # (n_drugs_total,)
        topk_indices = np.argsort(solution_score)[::-1][:topk]
        candidate_drugs = model.label_tokenizer.convert_indices_to_tokens(topk_indices)
        candidate_drugs = [code2drug[code] for code in candidate_drugs]

    elif fn == 'kg':
        target_nodes = [e for e in graph.entities if is_D(e)]
        target_nodes = [e for e in target_nodes if e in label_tokenizer.vocabulary.token2idx]
        target_node_ids = [graph.entity2id[e] for e in target_nodes]

        candidate_drugs = defaultdict(int)
        for c in conditions + procedures:
            triplets = get_neighbors(c, target_node_ids, direction='forward')
            for c, _, drug in triplets:
                candidate_drugs[drug] += 1

            triplets = get_neighbors(c, target_node_ids, direction='backward')
            for drug, _, c in triplets:
                candidate_drugs[drug] += 1

        candidate_drugs = sorted(candidate_drugs.items(), key=lambda x: x[1], reverse=True)
        candidate_drugs = [item[0] for item in candidate_drugs][:topk]

    elif fn == 'all':
        candidate_drugs = list(label_tokenizer.vocabulary.token2idx.keys())

    else:
        raise ValueError()

    return candidate_drugs


def get_candidate_diagnosis(fn, model, label_tokenizer, concepts, topk=1000):
    conditions, procedures, drugs = concepts
    if fn == 'dl':
        conditions, procedures, drugs = padding(conditions), padding(procedures), padding(drugs)
        kwargs = {'conditions': [conditions], 'procedures': [procedures], 'drugs': [drugs], 'label': [conditions[-1]]}
        with torch.no_grad():
            output = model(**kwargs)

        solution_score = output['y_prob'][0].sigmoid().cpu().numpy()  # (n_condition_total,)
        topk_indices = np.argsort(solution_score)[::-1][:topk]
        candidate_conditions = label_tokenizer.convert_indices_to_tokens(topk_indices)

    elif fn == 'kg':
        target_nodes = [e for e in graph.entities if is_C(e)]
        target_node_ids = [graph.entity2id[e] for e in target_nodes]

        candidate_conditions = defaultdict(int)
        for c in conditions + procedures + drugs:
            triplets = get_neighbors(c, target_node_ids, direction='forward')
            for c, _, cond in triplets:
                if edge_rule_filtering(c, _, cond):
                    candidate_conditions[cond] += 1

            triplets = get_neighbors(c, target_node_ids, direction='backward')
            for cond, _, c in triplets:
                if edge_rule_filtering(cond, _, c):
                    candidate_conditions[cond] += 1

        candidate_conditions = sorted(candidate_conditions.items(), key=lambda x: x[1], reverse=True)
        candidate_conditions = [item[0] for item in candidate_conditions][:topk]

    elif fn == 'all':
        candidate_conditions = list(label_tokenizer.vocabulary.token2id.values())

    else:
        candidate_conditions = []

    return candidate_conditions
