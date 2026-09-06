import os
import json
import shap
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import torch.nn.functional as F
from pyhealth.datasets import SampleEHRDataset

from data.dataloader import load_mappings
from graph.embedding.embedding import generate_embeddings
from modules.context import get_concepts
from modules.post_process import POST_PROCESS_FN, extract_json_blocks
from modules.candidate import (load_mortality_candidate_model, load_readmission_candidate_model,
                               load_lenofstay_candidate_model, load_drugrec_candidate_model,
                               load_diagnosis_candidate_model, code2drug)

from baselines.CoT.CoT import CoT
from baselines.CoT.prompt import TASK_DESCRIPTION, OUTPUT_INSTRUCTION, OUTPUT_FORMAT, TEMPLATE_RAG
from baselines.RetCare.template import TEMPLATE_KEYWORD_SYSTEM, TEMPLATE_KEYWORD_USER


class RetCare(CoT):
    """
    RetCare: Towards Interpretable Clinical Decision Making through LLM-Driven Medical Knowledge Retrieval
    https://openreview.net/forum?id=jqo1vk63qE
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

        self.task = config['task']
        if self.task == 'mortality':
            self.dl_model = load_mortality_candidate_model(config['dataset'], self.task, self.dataset)
        elif self.task == 'readmission':
            self.dl_model = load_readmission_candidate_model(config['dataset'], self.task, self.dataset)
        elif self.task == 'lenofstay':
            self.dl_model = load_lenofstay_candidate_model(config['dataset'], self.task, self.dataset)
        elif self.task == 'drugrec':
            self.dl_model = load_drugrec_candidate_model(config['dataset'], self.task, self.dataset, "StageNet")
        elif self.task == 'diagnosis':
            self.dl_model = load_diagnosis_candidate_model(config['dataset'], self.task, self.dataset, 'StageNet')
        else:
            raise ValueError

        all_cond, all_proc, all_drug = load_mappings()
        if self.task == 'mortality':
            self.target_nodes = [e for e in self.graph.entities if 'mortality' in e.lower()]
        elif self.task == 'readmission':
            self.target_nodes = [e for e in self.graph.entities if 'readmission' in e.lower() or 'readmit' in e.lower()]
        elif self.task == 'lenofstay':
            self.target_nodes = [e for e in self.graph.entities if 'stay' in e.lower()]
        elif self.task == 'drugrec':
            self.target_nodes = [e for e in all_drug.values() if e in self.graph.entities]
        elif self.task == 'diagnosis':
            self.target_nodes = [e for e in all_cond.values() if e in self.graph.entities]
        else:
            raise ValueError()

        self.topk_knowledge = int(self.hyperparam['topk_knowledge'])

        # Compute SHAP feature importance
        self.patient_shap_importance = self.shap_feature_importance(kwargs['test_set'])

        self.kg_embedding_file = os.path.join("./graph/embedding/triplet_embedding.pkl")
        self.triplets, self.triplet_embeddings = self.encode_knowledge_base()

        self.triplet_embeddings = torch.from_numpy(self.triplet_embeddings)

    def encode_knowledge_base(self):
        if os.path.exists(self.kg_embedding_file):
            with open(self.kg_embedding_file, 'rb') as f:
                triplet_embedding_dict = pickle.load(f)
        else:
            triplets = []
            for eid, (src, dst) in enumerate(self.graph.graph.edge_index.t().tolist()):
                relation = self.graph.id2relation[self.graph.graph.relation[eid].item()].replace('_', ' ').lower()
                src = self.graph.id2entity[src].lower()
                dst = self.graph.id2entity[dst].lower()
                triplet = f"{src} {relation} {dst}"
                triplets.append(triplet)

            triplet_embedding_dict = generate_embeddings(triplets, dimension=self.graph.entity_embedding.shape[-1], num_threads=16)

            with open(self.kg_embedding_file, 'wb') as f:
                pickle.dump(triplet_embedding_dict, f)

        triplets, triplet_embeddings = [], []
        for triplet, embedding in triplet_embedding_dict.items():
            triplets.append(triplet)
            triplet_embeddings.append(embedding)
        return triplets, np.stack(triplet_embeddings)

    def shap_feature_importance(self, test_set):
        print("Generating SHAP feature importance")
        shap_cache_file = Path(os.path.join(os.path.dirname(self.cache_dir), "shap.json"))
        if shap_cache_file.exists():
            return json.loads(shap_cache_file.read_text(encoding='utf-8'))

        else:
            feature_tk = {feature_key: self.dl_model.feat_tokenizers[feature_key] for feature_key in self.dl_model.feature_keys}

            start_idx = {"conditions": 0,
                         "procedures": feature_tk['conditions'].get_vocabulary_size(),
                         "drugs": feature_tk['conditions'].get_vocabulary_size() + feature_tk['procedures'].get_vocabulary_size()}
            end_idx = {feature_key: start_idx[feature_key] + feature_tk[feature_key].get_vocabulary_size()
                       for feature_key in self.dl_model.feature_keys}

            def encode(patient):
                conditions, procedures, drugs = get_concepts(patient, self.task, with_history=True, flatten=True)
                features = {"conditions": conditions, "procedures": procedures, "drugs": drugs}

                vector = np.zeros((sum(feature_tk[feature_key].get_vocabulary_size() for feature_key in self.dl_model.feature_keys),))
                for feature_key in self.dl_model.feature_keys:
                    for feature in features[feature_key]:
                        vector[start_idx[feature_key] + feature_tk[feature_key].vocabulary.token2idx[feature]] = 1.0

                return vector

            def decode(vector):
                features = {feature_key: [] for feature_key in self.dl_model.feature_keys}
                for feature_key in self.dl_model.feature_keys:
                    for idx in np.where(vector[start_idx[feature_key]:end_idx[feature_key]])[0]:
                        features[feature_key].append(feature_tk[feature_key].vocabulary.idx2token[idx])

                assert sum([len(values) for values in features.values()]) > 0
                return {feature_key: [values] for feature_key, values in features.items()}

            def prediction_wrapper(x):
                valid_mask = x.sum(axis=1) != 0

                all_output = np.zeros((len(x), self.dl_model.get_output_size(self.dl_model.label_tokenizer)))

                batch_size = 8192
                for i in range(0, len(x), batch_size):
                    batch_x = x[i:i + batch_size]
                    batch_valid_mask = valid_mask[i: i + batch_size]
                    batch_x = batch_x[batch_valid_mask]

                    decoded = [decode(vec) for vec in batch_x]
                    kwargs = {feature_key: [d[feature_key] for d in decoded] for feature_key in self.dl_model.feature_keys}
                    kwargs[self.dl_model.label_key] = [[]] * len(batch_x)

                    with torch.no_grad():
                        output = self.dl_model(**kwargs)
                        batch_output = output['y_prob'].cpu().numpy()

                    all_output[i:i + batch_size][batch_valid_mask] = batch_output

                return all_output

            encoded_testset = np.array([encode(p) for p in test_set])
            masker = shap.maskers.Independent(data=encoded_testset)
            explainer = shap.Explainer(prediction_wrapper, masker, seed=42, max_evals=2000)
            shap_values = explainer(encoded_testset)

            patient_shap_importance = {}
            for i in range(len(test_set)):
                sample_id = test_set[i]['patient_id'] + "_" + test_set[i]['visit_id']

                sample_explanation = {}
                conditions, procedures, drugs = get_concepts(test_set[i], self.task, with_history=True, flatten=True)
                features = {"conditions": conditions, "procedures": procedures, "drugs": drugs}

                token_dict = self.dl_model.label_tokenizer.vocabulary.idx2token
                for k, token in token_dict.items():
                    feature2value = {}
                    for feature_key in self.dl_model.feature_keys:
                        for feature in features[feature_key]:
                            shap_value = shap_values.values[i, start_idx[feature_key] + feature_tk[feature_key].vocabulary.token2idx[feature], k]
                            feature2value[feature] = shap_value

                    sample_explanation[token] = feature2value

                patient_shap_importance[sample_id] = sample_explanation

            shap_cache_file.write_text(json.dumps(patient_shap_importance), encoding='utf-8')
            return patient_shap_importance

    def get_keywords(self, sample_id, patient, similar_patients, candidates):
        # Section 4.2
        conditions, procedures, drugs = get_concepts(patient, self.task, with_history=True, flatten=True)
        features = {'conditions': conditions, 'procedures': procedures, 'drugs': drugs}

        features_importance = {}
        for feature_key in self.dl_model.feature_keys:
            for feature in features[feature_key]:
                feature_importance = {}
                for candidate in candidates:
                    imp = self.patient_shap_importance[sample_id][candidate][feature] * 2
                    if abs(imp) >= 0.1:
                        feature_importance[candidate] = round(imp, 3)

                features_importance[feature] = json.dumps(feature_importance)

        if self.n_shots > 0 and similar_patients is not None:
            few_shots = self.format_fewshot_examples(similar_patients)
        else:
            few_shots = ""

        context = "# Patient EHR Context #\n" + str(self.contexts[sample_id]) + "\n" + \
                  "# Patient's Feature-Candidate Correlation (out of 1.0) #\n" + "\n".join(f"{f}: {s}" for f, s in features_importance.items()) + \
                  few_shots
        prompt = TEMPLATE_KEYWORD_USER.replace("##CONTEXT_PLACEHOLDER##", context)

        def post_process_keyword(prob, response, output):
            json_obj = extract_json_blocks(response)
            keywords = list(json_obj["keywords"])
            return prob, (response, keywords), output

        _, (response, keywords), _, usage = self.model.understand(prompt,
                                                                  system=TEMPLATE_KEYWORD_SYSTEM,
                                                                  post_process=post_process_keyword)
        return response, keywords, usage

    def retrieve(self, sample_id, patient, similar_patients, candidates):
        def similarity(keyword_embeddings):
            batch_size = 1024
            sim_matrix = []
            for i in range(0, len(self.triplets), batch_size):
                batch_triplets = self.triplet_embeddings[i:i + batch_size].float().to(self.device).unsqueeze(0)
                batch_sim = F.cosine_similarity(keyword_embeddings, batch_triplets, dim=2)
                sim_matrix.append(batch_sim)

            return torch.cat(sim_matrix, dim=1)  # shape: (n_keywords, n_triplets)

        response, keywords, usage = self.get_keywords(sample_id, patient, similar_patients, candidates)
        keyword_embeddings = generate_embeddings(keywords, dimension=self.graph.entity_embedding.shape[-1])

        keyword_embeddings = [keyword_embeddings[word] for word in keywords]
        keyword_embeddings = torch.tensor(keyword_embeddings).unsqueeze(1).to(self.device)

        sim_matrix = similarity(keyword_embeddings)
        flat_sim_matrix = sim_matrix.flatten()
        topk_values, topk_indices_flat = torch.topk(flat_sim_matrix, k=self.topk_knowledge)
        topk_triplet_indices = topk_indices_flat % len(self.triplets)

        unique_triplet_indices = torch.unique(topk_triplet_indices)
        selected_knowledge = [self.triplets[idx.item()] for idx in unique_triplet_indices]

        return response, selected_knowledge, usage

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        _, medical_knowledge, usage = self.retrieve(sample_id, patient,
                                                    similar_patients=similar_patient,
                                                    candidates=candidates)
        medical_knowledge = "\n".join(medical_knowledge)

        if candidates is not None and self.candidate_fn is not None:
            patient_context += self.format_candidates(candidates)

        few_shots = self.format_fewshot_examples(similar_patient)

        prompt = TEMPLATE_RAG \
            .replace("##TASK_PLACEHOLDER##", TASK_DESCRIPTION[self.task]) \
            .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
            .replace("##FEWSHOT_PLACEHOLDER##", few_shots) \
            .replace("##KNOWLEDGE_PLACEHOLDER##", medical_knowledge) \
            .replace("##INSTRUCTION_PLACEHOLDER##", OUTPUT_INSTRUCTION[self.task]) \
            .replace("##FORMAT_PLACEHOLDER##", OUTPUT_FORMAT[self.task])

        _, (response, prediction), _, _usage = self.model.understand(query=prompt, post_process=POST_PROCESS_FN[self.task])
        usage += _usage
        return {"query": prompt, "response": response, "prediction": prediction, "usage": usage.dict()}
