import random
from typing import List, Dict
from pyhealth.datasets import SampleEHRDataset
from modules.context import get_concepts
from modules.post_process import POST_PROCESS_FN
from modules.kg import get_neighbors

from baselines.CoT.prompt import TEMPLATE_RAG, TASK_DESCRIPTION, OUTPUT_INSTRUCTION, OUTPUT_FORMAT
from baselines.CoT.CoT import CoT


class MedRAG(CoT):
    """
    MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot
    http://arxiv.org/abs/2502.04413
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
        super(MedRAG, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            config=config,
            **kwargs
        )

        self.n_knowledge = self.hyperparam["n_knowledge"]

    def retrieve(self, patient, candidates=None):
        conditions, procedures, drugs = get_concepts(patient, self.task, with_history=False, flatten=True)

        if self.task in ['drugrec']:
            drugs += candidates
        elif self.task in ['diagnosis']:
            conditions += candidates

        # retrieve from KG, modified from MedRAG
        condition_nodes = [concept for concept in conditions if concept in self.graph.entities]
        procedure_nodes = [concept for concept in procedures if concept in self.graph.entities]
        drug_nodes = [concept for concept in drugs if concept in self.graph.entities]

        subgraph = []
        for condition in condition_nodes:
            triples = get_neighbors(condition, procedure_nodes + drug_nodes, direction='both')
            subgraph.extend(triples)

        subgraph = [f"{e[0]} " + e[1].replace("_", "").lower() + f" {e[2]}" for e in subgraph]
        if len(subgraph) > self.n_knowledge:
            subgraph = random.choices(subgraph, k=self.n_knowledge)
        return subgraph

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        subgraph = self.retrieve(patient, candidates=candidates)
        medical_knowledge = "\n".join(subgraph)

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

        _, (response, prediction), _, usage = self.model.understand(query=prompt, post_process=POST_PROCESS_FN[self.task])
        return {"query": prompt, "response": response, "prediction": prediction, "usage": usage.dict()}
