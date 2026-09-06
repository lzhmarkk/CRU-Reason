import os
import torch
import json
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from pyhealth.datasets import SampleEHRDataset
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import to_networkx

from utils.utils import Usage
from data.dataloader import load_mappings
from modules.context import get_concepts
from modules.kg import graph, get_triplet
from graph.embedding.embedding import generate_embeddings

from baselines.CoT.CoT import CoT
from baselines.MedIKAL.doctor import Doctor


class MedIKAL(CoT):
    """
    medIKAL: Integrating Knowledge Graphs as Assistants of LLMs for Enhanced Clinical Diagnosis on EMRs
    http://arxiv.org/abs/2406.14326
    """

    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            config: Dict,
            **kwargs
    ):
        super().__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            config=config,
            **kwargs
        )

        self.doctor = Doctor(self.model, self.task)
        self.chunk_size = 1000
        self.direct_topn = self.hyperparam['direct_topn']

        all_cond, all_proc, all_drug = load_mappings()
        if self.task == 'mortality':
            self.target_nodes = [e for e in graph.entities if 'mortality' in e.lower()]
        elif self.task == 'readmission':
            self.target_nodes = [e for e in graph.entities if 'readmission' in e.lower() or 'readmit' in e.lower()]
        elif self.task == 'lenofstay':
            self.target_nodes = [e for e in graph.entities if 'stay' in e.lower()]
        elif self.task == 'drugrec':
            self.target_nodes = [e for e in all_drug.values() if e in graph.entities]
        elif self.task == 'diagnosis':
            self.target_nodes = [e for e in all_cond.values() if e in graph.entities]
        else:
            raise ValueError()

        self.path_cache_dir = os.path.join("./cache/shared/MedIKAL")
        os.makedirs(self.path_cache_dir, exist_ok=True)
        self.direct_cache_dir = os.path.join(self.cache_dir, 'direct')
        os.makedirs(self.direct_cache_dir, exist_ok=True)

        self.pairwise_paths = self.generate_shortest_path()

        self.graph = graph  # un-clustered graph

    def generate_shortest_path(self):
        # identical to baselines.MedReason
        cache_file = os.path.join(self.path_cache_dir, f"{self.task}_paths.json")

        if os.path.exists(cache_file):
            pairwise_paths = json.loads(Path(cache_file).read_text(encoding='utf-8'))

        else:
            graph_nx = to_networkx(graph.graph)

            all_cond, all_proc, all_drug = load_mappings()
            all_concepts = list(all_cond.values()) + list(all_proc.values()) + list(all_drug.values())

            pairwise_paths = {}
            for source in tqdm(all_concepts):
                if source not in graph.entities: continue
                for destination in self.target_nodes:
                    if destination not in graph.entities: continue
                    try:
                        paths = list(nx.all_shortest_paths(graph_nx, graph.entity2id[source], graph.entity2id[destination]))  # List[List[str]]
                    except:
                        continue

                    pairwise_paths[f"{source}->{destination}"] = paths

            Path(cache_file).write_text(json.dumps(pairwise_paths), encoding='utf-8')

        return pairwise_paths

    def find_nearest_entity(self, candidates, min_sim=0.8):
        candidate_embedding_dict = generate_embeddings(candidates, num_threads=8, dimension=1024)
        candidate_embeddings = [candidate_embedding_dict[item] for item in candidates]

        possible_entities = list(self.label_tokenizer.vocabulary.token2idx.keys())
        possible_entity_embeddings = [graph.entity2embedding[item.lower()] if item.lower() in graph.entities else torch.zeros(1024) \
                                      for item in possible_entities]

        sim_matrix = cosine_similarity(candidate_embeddings, possible_entity_embeddings)
        nearest_entity_id = np.argmax(sim_matrix, axis=1)
        nearest_entity_sim = np.max(sim_matrix, axis=1)
        nearest_entities = [possible_entities[i] for i, sim in zip(nearest_entity_id, nearest_entity_sim) if sim >= min_sim]

        return nearest_entities

    def rerank_by_path(self, targets, context):
        target_scores = defaultdict(list)
        target_paths = defaultdict(lambda: defaultdict(list))

        for source in context:
            # if source not in graph.entities: continue
            for target in targets:
                if source not in graph.entities or target not in graph.entities: continue
                paths = self.pairwise_paths.get(f"{source}->{target}", [])  # List[List[str]]
                for path in paths:
                    elements = [graph.id2entity[path[0]]]
                    for src, dst in zip(path[:-1], path[1:]):
                        src_name, edge_name, dst_name = get_triplet(src, dst)
                        if src_name is None or edge_name is None or dst_name is None: continue
                        elements.append(edge_name)
                        elements.append(dst_name)

                    target_paths[target][source].append(elements)
                    target_scores[target].append((len(elements) - 1) // 2)  # score

        # use mean instead of sum, which is more friendly for direct-connection
        target_scores = {target: (1 / np.array(score_list)).mean() for target, score_list in target_scores.items()}
        candidates_sorted = [e[0] for e in sorted(target_scores.items(), key=lambda x: x[1], reverse=True)]

        return candidates_sorted, dict(target_paths), dict(target_scores)

    def get_direct_prediction(self, sample_id, patient):
        cache_file = Path(os.path.join(self.direct_cache_dir, f"{sample_id}.json"))
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding='utf-8'))
            direct_prediction, usage = cached["direct_prediction"], Usage(cached["usage"])

        else:
            fst_prompt, fst_rd_summary, usage = self.doctor.general_info_summary(patient)

            scd_prompt, scd_rd_summary, _usage = self.doctor.examination_summary(patient)
            usage += _usage

            direct_prompt, direct_reasoning_response, direct_reasoning_result, _usage = \
                self.doctor.direct_reason(fst_rd_summary, scd_rd_summary, self.direct_topn)
            usage += _usage

            direct_prediction = self.find_nearest_entity(direct_reasoning_result)

            cached = {
                "fst_prompt": fst_prompt,
                "fst_response": fst_rd_summary,
                "scd_prompt": scd_prompt,
                "scd_response": scd_rd_summary,
                "direct_prompt": direct_prompt,
                "direct_response": direct_reasoning_response,
                "direct_prediction": direct_prediction,
                "usage": usage.dict()
            }
            cache_file.write_text(json.dumps(cached, indent=4), encoding='utf-8')

        return direct_prediction, usage

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        direct_prediction, usage = self.get_direct_prediction(sample_id, patient)

        candidates = direct_prediction + candidates

        conditions, procedures, drugs = get_concepts(patient, self.task, with_history=True, flatten=True)
        context = conditions + procedures + drugs

        candidates_sorted, candidate_paths, candidate_scores = self.rerank_by_path(candidates, context)

        prompt, response, prediction = [], [], {}
        for i, candidate in enumerate(candidates_sorted):
            cur_formatted_knowledge = ""
            cur_formatted_knowledge += f"# Candidate: {candidate}\n"

            for feature in ['conditions', 'procedures']:
                # Relation with patients' historical disease/check
                # "entail", "strong/weak", "irrelevant"
                cur_formatted_knowledge += f"## Connection with patient's {feature}\n"
                for concept, paths in candidate_paths[candidate].items():
                    if concept in {"conditions": conditions, "procedures": procedures}[feature]:
                        cur_formatted_knowledge += f"{concept}:"

                        if concept == candidate:
                            cur_formatted_knowledge += " Entail"
                        else:
                            min_length = min((len(path) - 1) // 2 for path in paths)
                            paths = random.sample(paths, 8) if len(paths) > 8 else paths
                            path_str = "\n".join("->".join(path) for path in paths) + "\n\n"
                            if min_length == 1:
                                cur_formatted_knowledge += f" Strong Correlation\n{path_str}"
                            elif min_length == 2:
                                cur_formatted_knowledge += f" Weakly Correlation\n{path_str}"
                            else:
                                cur_formatted_knowledge += " Irrelevant"

            _prompt, _response, score, _usage = self.doctor.analysis(candidate, cur_formatted_knowledge, conditions, procedures)
            prompt.append(_prompt)
            response.append(_response)
            prediction[candidate] = {"reason": _response, "score": score}
            usage += _usage

        return {"query": prompt, "response": response, "prediction": prediction, "usage": usage.dict()}
