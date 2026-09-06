import json
import random
from typing import List, Dict
from pyhealth.datasets import SampleEHRDataset

from graph.graphloader import load_graph

from baselines.CoT.context.base_context import get_patient_embedding
from baselines.KARE.dataset.context import KARE_context

from finetune.CoT.CoT import CoTFineTune
from baselines.KARE.prompt import KARE_TEMPLATE as KARE_INFERENCE_TEMPLATE
from baselines.CoT.prompt import TASK_DESCRIPTION, OUTPUT_INSTRUCTION, OUTPUT_FORMAT
from finetune.CoT.prompt import FINETUNE_OUTPUT_INSTRUCTION, FINETUNE_OUTPUT_FORMAT
from finetune.KARE.prompt import KARE_FINETUNE_TEMPLATE
from finetune.KARE.post_process import POST_PROCESS_FN


class KAREFineTune(CoTFineTune):
    """
    Reasoning-Enhanced Healthcare Predictions with Knowledge Graph Community Retrieval
    http://arxiv.org/abs/2410.04585
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

        self.graph = load_graph(cluster=True)

        # input context
        _, train_embeddings, _, _ = get_patient_embedding(kwargs['train_set'], task=self.task,
                                                          output_dir=f"./cache/{config['dataset']}-{config['task']}/CoT/train/")
        train_selected_summaries = KARE_context(sample_dataset=kwargs['train_set'],
                                                embeddings=train_embeddings, g=self.graph, model=self.model, task=self.task,
                                                output_dir=f"./cache/{config['dataset']}-{config['task']}/KARE/train/")

        _, val_embeddings, _, _ = get_patient_embedding(kwargs['val_set'], task=self.task,
                                                        output_dir=f"./cache/{config['dataset']}-{config['task']}/CoT/val/")
        val_selected_summaries = KARE_context(sample_dataset=kwargs['val_set'],
                                              embeddings=val_embeddings, g=self.graph, model=self.model, task=self.task,
                                              output_dir=f"./cache/{config['dataset']}-{config['task']}/KARE/val/")

        self.selected_summaries = train_selected_summaries | val_selected_summaries

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        medical_knowledge = self.selected_summaries[sample_id]
        medical_knowledge = [f"{id + 1}: {knowledge}" for id, knowledge in enumerate(medical_knowledge)]
        medical_knowledge = "\n".join(k.replace("\n", "") for k in medical_knowledge)

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

        prompt = KARE_FINETUNE_TEMPLATE \
            .replace("##TASK_PLACEHOLDER##", TASK_DESCRIPTION[self.task]) \
            .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
            .replace("##FEWSHOT_PLACEHOLDER##", few_shots) \
            .replace("##KNOWLEDGE_PLACEHOLDER##", medical_knowledge) \
            .replace("##TRUTH_PLACEHOLDER##", json.dumps(shuffled_elements, indent=2)) \
            .replace("##INSTRUCTION_PLACEHOLDER##", FINETUNE_OUTPUT_INSTRUCTION[self.task]).replace("4. P", "5. P") \
            .replace("##FORMAT_PLACEHOLDER##", FINETUNE_OUTPUT_FORMAT[self.task]).replace("4. P", "5. P")  # update index 4->5

        _, response, _, usage = self.model.understand(query=prompt, post_process=POST_PROCESS_FN[self.task])
        response.update({"query": prompt, "patient_label": ground_truth, "pos_label": positive_elements,
                         "neg_label": negative_elements, "usage": usage.dict()})
        return response

    def construct_input_output(self, sample_id, patient, patient_context, sample, similar_patient=None):
        medical_knowledge = self.selected_summaries[sample_id]
        medical_knowledge = [f"{id + 1}: {knowledge}" for id, knowledge in enumerate(medical_knowledge)]
        medical_knowledge = "\n".join(k.replace("\n", "") for k in medical_knowledge)

        few_shots = self.format_fewshot_examples(similar_patient)

        candidates = list(sample['prediction'].keys())
        patient_context += self.format_candidates(candidates)

        prompt = KARE_INFERENCE_TEMPLATE \
            .replace("##TASK_PLACEHOLDER##", TASK_DESCRIPTION[self.task]) \
            .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
            .replace("##FEWSHOT_PLACEHOLDER##", few_shots) \
            .replace("##KNOWLEDGE_PLACEHOLDER##", medical_knowledge) \
            .replace("##INSTRUCTION_PLACEHOLDER##", OUTPUT_INSTRUCTION[self.task]).replace("4. P", "5. P") \
            .replace("##FORMAT_PLACEHOLDER##", OUTPUT_FORMAT[self.task]).replace("4. P", "5. P")  # update index 4->5

        prediction = "```json\n" + json.dumps(sample['prediction'], indent=2) + "\n```"
        response = f"""\
1. Overview:\n{sample['overview']}
2. Relevant Knowledge:\n{sample['knowledge']}
3. Comparison:\n{sample['comparison']}
4. Conclusion:\n{sample['reasoning']}
5. Prediction:\n{prediction}
"""

        return prompt, response
