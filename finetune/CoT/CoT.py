import os
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from pyhealth.datasets import SampleEHRDataset

from utils.llm import load_model
from graph.graphloader import load_graph
from utils.multiprocess import multiprocess_call

from baselines.CoT.CoT import CoT
from baselines.CoT.prompt import TASK_DESCRIPTION, OUTPUT_INSTRUCTION, OUTPUT_FORMAT
from baselines.CoT.prompt import TEMPLATE as INFERENCE_TEMPLATE
from baselines.CoT.utils import get_similar_patient_label
from baselines.CoT.context import CoT_context
from finetune.CoT.prompt import FINETUNE_OUTPUT_INSTRUCTION, FINETUNE_OUTPUT_FORMAT, FINETUNE_TEMPLATE
from finetune.CoT.post_process import POST_PROCESS_FN


class CoTFineTune(CoT):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            config: Dict,
            **kwargs
    ):
        super(CoT, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode
        )  # skip baselines.CoT.__init__, directly run pyhealth.BaseModel.__init__

        # hyper-parameters
        self.hyperparam = json.loads(Path(f"./finetune/{config['method']}/hyperparam.json").read_text(encoding='utf-8'))
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
        train_contexts, train_similar_patients = CoT_context(sample_dataset=kwargs['train_set'],
                                                             task=self.task,
                                                             output_dir=f"./cache/{config['dataset']}-{config['task']}/CoT/train/")
        val_contexts, val_similar_patients = CoT_context(sample_dataset=kwargs['val_set'],
                                                         task=self.task,
                                                         output_dir=f"./cache/{config['dataset']}-{config['task']}/CoT/val/")

        self.contexts = train_contexts | val_contexts  # dict
        self.similar_patients = train_similar_patients | val_similar_patients  # dict

        # evaluate every 20 candidates to limit input context length
        self.use_chunk = self.task in ['drugrec', 'diagnosis']
        self.chunk_size = 20
        self.use_chunk = False

        self.cache_parent_dir = f"./cache/{config['dataset']}-{config['task']}/{config['method']}"
        self.cache_dirs, self.reason_cache_dirs = {}, {}
        for split in ['train', 'val']:
            self.cache_dirs[split] = os.path.join(self.cache_parent_dir, split)
            self.reason_cache_dirs[split] = os.path.join(self.cache_dirs[split], "reason")
            os.makedirs(self.reason_cache_dirs[split], exist_ok=True)

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        few_shots = self.format_fewshot_examples(similar_patient)

        # select chunk_size/2 positive candidates and chunk_size/2 negative candidates
        random.seed(sample_id)
        ground_truth = patient[self.label_key]
        rejected_truth = [drug for drug in self.label_tokenizer.vocabulary.token2idx.keys() if drug not in ground_truth]
        positive_elements = random.sample(ground_truth, min(self.chunk_size // 2, len(ground_truth)))
        negative_elements = random.sample(rejected_truth, min(self.chunk_size // 2, len(rejected_truth)))
        elements = {}
        for element in positive_elements:
            elements[element] = {"score": random.choice([3, 4])}  # assign relatively high scores for positives
        for element in negative_elements:
            elements[element] = {"score": random.choice([0, 1])}  # assign relatively low scores for positives

        shuffled_keys = random.sample(list(elements.keys()), len(elements))
        shuffled_elements = {k: elements[k] for k in shuffled_keys}

        prompt = FINETUNE_TEMPLATE \
            .replace("##TASK_PLACEHOLDER##", TASK_DESCRIPTION[self.task]) \
            .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
            .replace("##FEWSHOT_PLACEHOLDER##", few_shots) \
            .replace("##TRUTH_PLACEHOLDER##", json.dumps(shuffled_elements, indent=2)) \
            .replace("##INSTRUCTION_PLACEHOLDER##", FINETUNE_OUTPUT_INSTRUCTION[self.task]) \
            .replace("##FORMAT_PLACEHOLDER##", FINETUNE_OUTPUT_FORMAT[self.task])

        _, response, _, usage = self.model.understand(query=prompt, post_process=POST_PROCESS_FN[self.task])
        response.update({"query": prompt, "patient_label": ground_truth, "pos_label": positive_elements,
                         "neg_label": negative_elements, "usage": usage.dict()})
        return response

    def construct_input_output(self, sample_id, patient, patient_context, sample, similar_patient=None):
        few_shots = self.format_fewshot_examples(similar_patient)

        candidates = list(sample['prediction'].keys())
        patient_context += self.format_candidates(candidates)

        prompt = INFERENCE_TEMPLATE \
            .replace("##TASK_PLACEHOLDER##", TASK_DESCRIPTION[self.task]) \
            .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
            .replace("##FEWSHOT_PLACEHOLDER##", few_shots) \
            .replace("##INSTRUCTION_PLACEHOLDER##", OUTPUT_INSTRUCTION[self.task]) \
            .replace("##FORMAT_PLACEHOLDER##", OUTPUT_FORMAT[self.task])

        prediction = "```json\n" + json.dumps(sample['prediction'], indent=2) + "\n```"
        response = f"""\
1. Overview:\n{sample['overview']}
2. Comparison:\n{sample['comparison']}
3. Conclusion:\n{sample['reasoning']}
4. Prediction:\n{prediction}
"""

        return prompt, response

    def generate_dataset(self, **kwargs) -> List[Dict[str, Any]]:
        # kwargs: {'visit_id': List[str], 'patient_id': List[str], 'conditions': List[List[List[str]]]], 'label': List[int]}
        split = kwargs['split']
        assert split in ['train', 'val', 'test']

        items = []
        for i in range(len(kwargs['conditions'])):
            sample_id = kwargs['patient_id'][i] + "_" + kwargs['visit_id'][i]
            patient = {key: kwargs[key][i] for key in self.feature_keys}
            patient[self.label_key] = kwargs[self.label_key][i]
            context = self.contexts[sample_id]

            similar_patient = self.similar_patients[sample_id] if sample_id in self.similar_patients else None

            items.append((sample_id, patient, context, similar_patient))

        def process_single(key, item):
            sample_id, patient, context, similar_patient = item

            output_file = os.path.join(self.reason_cache_dirs[split], f"{sample_id}.json")
            if os.path.exists(output_file):
                output = json.loads(Path(output_file).read_text(encoding='utf-8'))

            else:
                time.sleep(random.randint(5, 20))

                output = self.generate_reasoning(sample_id=sample_id,
                                                 patient=patient,
                                                 patient_context=context,
                                                 similar_patient=similar_patient)

                Path(output_file).write_text(json.dumps(output, indent=4), encoding='utf-8')

            confidence = output["confidence"]
            pos_label, neg_label = output["pos_label"], output['neg_label']

            if "Not Confident" not in confidence:  # low quality sample
                query, output = self.construct_input_output(sample_id=sample_id,
                                                            patient=patient,
                                                            patient_context=context,
                                                            sample=output,
                                                            similar_patient=similar_patient)
                return key, {"input": query, "output": output, 'pos_label': pos_label, 'neg_label': neg_label}

            else:
                return key, None

        samples = multiprocess_call(process_single, items, n_workers=self.n_workers)
        samples = [samples for samples in samples if samples is not None]
        return samples
