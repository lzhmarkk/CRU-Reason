import os
import time
import json
import torch
import random
from pathlib import Path
from typing import List, Dict, Any
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset

from utils.llm import load_model
from utils.utils import Usage
from utils.multiprocess import multiprocess_call
from graph.graphloader import load_graph

from modules.context import get_concepts
from modules.post_process import POST_PROCESS_FN, get_logit, logger
from modules.candidate import (load_lenofstay_candidate_model, load_drugrec_candidate_model, load_diagnosis_candidate_model,
                               get_candidate_los, get_candidate_drugs, get_candidate_diagnosis)

from baselines.CoT.prompt import timespans, TASK_DESCRIPTION, OUTPUT_INSTRUCTION, OUTPUT_FORMAT, TEMPLATE
from baselines.CoT.utils import get_similar_patient_label
from baselines.CoT.context import CoT_context


class CoT(BaseModel):
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
            mode=mode
        )

        # hyper-parameters
        self.hyperparam = json.loads(Path(f"./baselines/{config['method']}/hyperparam.json").read_text(encoding='utf-8'))
        self.n_shots = self.hyperparam['n_shots']
        print(f"Use {self.n_shots} shot setting")
        self.task = config['task']
        self.n_workers = config['n_workers']

        # modules
        self.model = load_model(config)
        self.graph = load_graph(cluster=False)
        self.patient_labels = get_similar_patient_label(self.dataset, self.task)
        self.label_tokenizer = self.get_label_tokenizer()

        # input context
        self.contexts, self.similar_patients = CoT_context(sample_dataset=kwargs['test_set'],
                                                           task=self.task,
                                                           output_dir=f"./cache/{config['dataset']}-{config['task']}/CoT/test/")

        # evaluate every 20 candidates to limit input context length
        self.use_chunk = self.task in ['drugrec', 'diagnosis']
        self.chunk_size = 20

        # candidates provides by small model or KG
        self.dl_model = None
        self.candidate_fn = self.hyperparam['candidate_fn'][self.task]
        if self.candidate_fn is not None:
            if self.task == 'drugrec' and self.candidate_fn == 'dl':
                self.dl_model = load_drugrec_candidate_model(dataset, self.task, self.dataset).to(self.device)
            elif self.task == 'lenofstay' and self.candidate_fn == 'dl':
                self.dl_model = load_lenofstay_candidate_model(dataset, self.task, self.dataset).to(self.device)
            elif self.task == 'diagnosis' and self.candidate_fn == 'dl':
                self.dl_model = load_diagnosis_candidate_model(dataset, self.task, self.dataset).to(self.device)
            config['exp'] += f"-{self.candidate_fn}"

        # cached LLM outputs
        self.cache_dir = f"./cache/{config['dataset']}-{config['task']}/{config['method']}/{self.n_shots}shot-{self.model.genre}-{config['exp']}"
        os.makedirs(self.cache_dir, exist_ok=True)

    def format_fewshot_examples(self, similar_patient):
        if self.n_shots > 0 and similar_patient is not None:
            similar_patients = similar_patient['positive'] + similar_patient['negative']
            random.shuffle(similar_patients)
            similar_patients = similar_patients[:self.n_shots]
            similar_patients = [f"{self.contexts[patient]}\nLabel:\n{self.patient_labels[patient]}" for patient in similar_patients]
            similar_patients = "\n\n".join(similar_patients)
            few_shots = f"""\n========================================\n# Similar Patients #\n{similar_patients}\n"""
        else:
            few_shots = ""

        return few_shots

    def format_candidates(self, candidates):
        if self.task == 'drugrec':
            return f"\n\n# Candidate Drugs #\n" + "\n".join(candidates)
        elif self.task == 'lenofstay':
            return f"\n\n# Candidate LOS #\n" + ", ".join(candidates)
        elif self.task == 'diagnosis':
            return f"\n\n# Candidate Diseases #\n" + "\n".join(candidates)
        return ""

    def get_candidates(self, patient):
        conditions, procedures, drugs = get_concepts(patient, self.task, with_history=True, flatten=True)

        if self.candidate_fn is None: return None
        if self.task == 'drugrec':
            concept_tuple = (conditions, procedures, patient['drugs'])
            candidates = get_candidate_drugs(self.candidate_fn, self.dl_model, self.label_tokenizer,
                                             concepts=concept_tuple, topk=1000)
            candidates = [x for x in candidates if x in self.label_tokenizer.vocabulary]

        elif self.task == 'lenofstay':
            concept_tuple = (conditions, procedures, None)
            candidate_score = get_candidate_los(self.candidate_fn, self.dl_model, self.label_tokenizer,
                                                concepts=concept_tuple)
            candidates = [ts for ts, s in zip(timespans, candidate_score) if s > 0]

        elif self.task == 'diagnosis':
            concept_tuple = (conditions, procedures, drugs)
            candidates = get_candidate_diagnosis(self.candidate_fn, self.dl_model, self.label_tokenizer,
                                                 concepts=concept_tuple)
            candidates = [x for x in candidates if x in self.label_tokenizer.vocabulary]

        else:
            candidates = None

        return candidates

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        if candidates is not None and self.candidate_fn is not None:
            patient_context += self.format_candidates(candidates)

        few_shots = self.format_fewshot_examples(similar_patient)

        prompt = TEMPLATE \
            .replace("##TASK_PLACEHOLDER##", TASK_DESCRIPTION[self.task]) \
            .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
            .replace("##FEWSHOT_PLACEHOLDER##", few_shots) \
            .replace("##INSTRUCTION_PLACEHOLDER##", OUTPUT_INSTRUCTION[self.task]) \
            .replace("##FORMAT_PLACEHOLDER##", OUTPUT_FORMAT[self.task])

        _, (response, prediction), _, usage = self.model.understand(query=prompt, post_process=POST_PROCESS_FN[self.task])
        return {"query": prompt, "response": response, "prediction": prediction, "usage": usage.dict()}

    def generate_reasoning_chunk(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        assert self.task in ['drugrec', 'diagnosis']
        assert isinstance(candidates, list)

        outputs = {'query': [], 'response': [], 'prediction': {}, 'usage': Usage()}
        for chunk_id in range(0, len(candidates), self.chunk_size):
            chunk_candidates = candidates[chunk_id:chunk_id + self.chunk_size]

            output = self.generate_reasoning(sample_id, patient, patient_context, similar_patient,
                                             candidates=chunk_candidates)
            outputs['query'].append(output['query'])
            outputs['response'].append(output['response'])
            outputs['prediction'].update(output['prediction'])
            outputs['usage'] += output['usage']

        return {"query": outputs["query"], "response": outputs["response"], "prediction": outputs["prediction"],
                "usage": outputs["usage"].dict()}

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # kwargs: {'visit_id': List[str], 'patient_id': List[str], 'conditions': List[List[List[str]]]], 'label': List[int]}
        items = []
        for i in range(len(kwargs['conditions'])):
            sample_id = kwargs['patient_id'][i] + "_" + kwargs['visit_id'][i]
            patient = {key: kwargs[key][i] for key in self.feature_keys}
            if 'drugs' not in self.feature_keys:
                patient['drugs'] = kwargs['drugs'][i]  # for drugrec placeholders
            context = self.contexts[sample_id]

            similar_patient = self.similar_patients[sample_id] if sample_id in self.similar_patients else None

            items.append((sample_id, patient, context, similar_patient))

        def process_single(key, item):
            sample_id, patient, context, similar_patient = item

            output_file = os.path.join(self.cache_dir, f"{sample_id}.json")
            if os.path.exists(output_file):
                output = json.loads(Path(output_file).read_text(encoding='utf-8'))

            else:
                time.sleep(random.randint(0, int(self.n_workers // 2)))
                candidates = self.get_candidates(patient)

                if self.use_chunk:
                    output = self.generate_reasoning_chunk(sample_id=sample_id,
                                                           patient=patient,
                                                           patient_context=context,
                                                           similar_patient=similar_patient,
                                                           candidates=candidates)
                else:
                    output = self.generate_reasoning(sample_id=sample_id,
                                                     patient=patient,
                                                     patient_context=context,
                                                     similar_patient=similar_patient,
                                                     candidates=candidates)

                Path(output_file).write_text(json.dumps(output, indent=4), encoding='utf-8')

            # output logits
            logger.info(f"******************** {sample_id} ********************")
            # logger.info(output["query"])
            # logger.info(output["response"])
            logit = get_logit(self.task, output["prediction"], self.label_tokenizer)
            return key, logit

        logits = multiprocess_call(process_single, items, n_workers=self.n_workers)
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        logits = torch.tensor(logits).to(y_true.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = logits
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        return results
