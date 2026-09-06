import os
import json
import random
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from pyhealth.datasets import SampleEHRDataset
from torch_geometric.utils import to_networkx

from data.dataloader import load_mappings
from utils.utils import Usage
from modules.context import get_concepts
from modules.post_process import POST_PROCESS_FN
from modules.kg import graph, get_triplet

from baselines.CoT.prompt import TEMPLATE_RAG, TASK_DESCRIPTION, OUTPUT_INSTRUCTION, OUTPUT_FORMAT
from baselines.CoT.CoT import CoT
from baselines.MedReason.prompt import REFORMAT_TEMPLATE
from baselines.MKGRank.prompts import TASK_DEFINITION


class MedReason(CoT):
    """
    MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs
    http://arxiv.org/abs/2504.00993
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
        super(MedReason, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            config=config,
            **kwargs
        )

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

        self.max_paths_per_group = self.hyperparam['max_paths_per_group']
        self.topK_reasoning_paths = self.hyperparam['topK_reasoning_paths']
        self.max_hops = self.hyperparam['max_hops']  # since too many (concept, target) pairs (>400 with 2 hops), we set max_hops to 1

        self.path_cache_dir = os.path.join("./cache/shared/MedReason")
        os.makedirs(self.path_cache_dir, exist_ok=True)
        self.pairwise_paths = self.generate_shortest_path()

    def generate_shortest_path(self):
        # identical to baselines.MedIKAL
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

    def retrieve(self, patient, candidates=None):
        if self.task in ['drugrec', 'diagnosis']:
            targets = candidates
        else:
            targets = self.target_nodes

        conditions, procedures, drugs = get_concepts(patient, self.task, with_history=False, flatten=True)
        concepts = conditions + procedures + drugs

        all_paths = []
        for source in concepts:
            if source not in graph.entities: continue
            for destination in targets:
                if destination not in graph.entities: continue
                paths = self.pairwise_paths.get(f"{source}->{destination}", [])  # List[List[str]]
                for path in paths:
                    skip = False
                    elements = [graph.id2entity[path[0]]]
                    for src, dst in zip(path[:-1], path[1:]):
                        src_name, edge_name, dst_name = get_triplet(src, dst)
                        assert src_name is not None and edge_name is not None and dst_name is not None
                        elements.append(edge_name)
                        elements.append(dst_name)

                        # if any(w in edge_name.lower() for w in negative_relation + vague_relation):
                        #     skip = True

                    if not skip and len(elements) <= self.max_hops * 2 + 1:
                        all_paths.append(elements)

        return all_paths

    def __most_correlated_path_selection_cached(self, path_key, path_values, patient_context):
        start, end = graph.entity2id[path_key[0]], graph.entity2id[path_key[1]]
        cache_file = os.path.join(self.path_cache_dir, f"{start}-{end}.json")
        if os.path.exists(cache_file):
            cached = json.loads(Path(cache_file).read_text(encoding='utf-8'))
            llm_selected_paths = cached['paths']
            usage = cached['usage']

        else:
            # if the number of paths in the group is more than max_path_number_per_group
            # randomly sample max_path_number_per_group paths from the group before use LLM to further sample
            if len(path_values) > self.max_paths_per_group:
                path_values = random.sample(path_values, self.max_paths_per_group)

            # get the sampled paths in json format
            def post_process(prob, response, output):
                start = response.find("{")
                end = response.rfind("}")
                json_str = response[start:end + 1]
                json_obj = json.loads(json_str)
                selected_paths = json_obj['Paths']
                return prob, selected_paths, output

            # convert the list of similar entities to a string
            text_for_group_paths = '\n'.join([str(idx + 1) + ': ' + '->'.join(triplet) for idx, triplet in enumerate(path_values)])
            prompt = REFORMAT_TEMPLATE \
                .replace("##TOPK_PLACEHOLDER##", str(self.topK_reasoning_paths)) \
                .replace("##TASK_PLACEHOLDER##", TASK_DEFINITION[self.task].split(":")[0]) \
                .replace("##DESCRIPTION_PLACEHOLDER##", TASK_DEFINITION[self.task]) \
                .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
                .replace("##PATH_PLACEHOLDER##", text_for_group_paths)

            _, llm_selected_paths, _, usage = self.model.understand(prompt, post_process=post_process)
            cached = {"paths": llm_selected_paths, "usage": usage.dict()}
            Path(cache_file).write_text(json.dumps(cached, indent=4), encoding='utf-8')

        return llm_selected_paths, usage

    def path_sampling(self, paths, patient_context):
        path_groups = defaultdict(list)
        for path in paths:
            if len(path) < 2:
                continue

            path_key = (path[0], path[-1])
            path_groups[path_key].append(path)

        # traverse the path_groups, and sample topK reasoning paths from each group
        sampled_paths = []
        usage = Usage()
        for path_key in path_groups:
            if len(path_groups[path_key]) <= self.topK_reasoning_paths:
                sampled_paths.extend(path_groups[path_key])

            else:
                result_for_group, _usage = self.__most_correlated_path_selection_cached(path_key, path_groups[path_key], patient_context)
                usage += _usage

                # for each path_i in result_for_group["Paths"], add path_i["path"] to sampled_paths
                for path_i in result_for_group:
                    sampled_paths.append(path_i["path"].split('->'))

        return sampled_paths, usage

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        path_all = self.retrieve(patient, candidates=candidates)
        path_all, usage = self.path_sampling(path_all, patient_context)
        output_text = '\n'.join([str(idx + 1) + ': ' + '->'.join(triplet) for idx, triplet in enumerate(path_all)])

        if candidates is not None and self.candidate_fn is not None:
            patient_context += self.format_candidates(candidates)

        few_shots = self.format_fewshot_examples(similar_patient)

        prompt = TEMPLATE_RAG \
            .replace("##TASK_PLACEHOLDER##", TASK_DESCRIPTION[self.task]) \
            .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
            .replace("##FEWSHOT_PLACEHOLDER##", few_shots) \
            .replace("##KNOWLEDGE_PLACEHOLDER##", output_text) \
            .replace("##INSTRUCTION_PLACEHOLDER##", OUTPUT_INSTRUCTION[self.task]) \
            .replace("##FORMAT_PLACEHOLDER##", OUTPUT_FORMAT[self.task])

        _, (response, prediction), _, _usage = self.model.understand(query=prompt, post_process=POST_PROCESS_FN[self.task])
        usage += _usage
        return {"query": prompt, "response": response, "prediction": prediction, "usage": usage.dict()}
