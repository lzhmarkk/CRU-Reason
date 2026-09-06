import nltk
import random
import itertools
from typing import List, Dict
from pyhealth.datasets import SampleEHRDataset

from utils.utils import Usage
from modules.context import get_concepts
from modules.post_process import POST_PROCESS_FN, parse_string

from baselines.CoT.prompt import TEMPLATE_RAG, TASK_DESCRIPTION, OUTPUT_INSTRUCTION, OUTPUT_FORMAT
from baselines.CoT.CoT import CoT
from baselines.MKGRank.context import generate_concepts_context
from baselines.MKGRank.umls import UMLSBERT, UMLS_CrossEncoder
from baselines.MKGRank.prompts import TASK_DEFINITION, TRIPLET_TO_SENTENCE_TEMPLATE


class MKGRank(CoT):
    """
    MKG-Rank: Enhancing Large Language Models with Knowledge Graph for Multilingual Medical Question Answering
    http://arxiv.org/abs/2503.16131
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
        super(MKGRank, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            config=config,
            **kwargs
        )

        self.umls_bert = UMLSBERT("NitzanBar/umls-bert")
        self.cross_encoder = UMLS_CrossEncoder("ncbi/MedCPT-Cross-Encoder")

        self.n_triplets_per_sample = self.hyperparam['n_triplets_per_sample']
        self.n_description_sentences = self.hyperparam['n_description_sentences']

        self.concept_description, self.concept_triplets, self.concept_description_usage = \
            generate_concepts_context(task=self.task,
                                      encoder=self.umls_bert,
                                      llm=self.model,
                                      cache_dir="./cache/shared/MKGRank")

    def retrieve(self, patient, candidates=None):
        context = list(itertools.chain.from_iterable(get_concepts(patient, self.task, with_history=True, flatten=True)))

        if candidates is None:
            targets = list(itertools.chain.from_iterable(get_concepts(patient, self.task, with_history=False, flatten=True)))
        else:
            targets = list(candidates)

        cut_description = lambda x: ' '.join(nltk.sent_tokenize(x)[:self.n_description_sentences])
        target_descriptions, usage = [], Usage()
        for concept in targets:
            description = cut_description(self.concept_description.get(concept, ""))
            target_descriptions.append(description)
            usage += self.concept_description_usage[concept]

        triplets = list(itertools.chain.from_iterable([self.concept_triplets.get(concept, []) for concept in context + targets]))
        triplets_str = ['(' + ', '.join(triplet) + ')' for triplet in triplets]
        scores = self.cross_encoder.score(TASK_DEFINITION[self.task], triplets_str)
        scored_triples = list(zip(scores, triplets))
        scored_triples.sort(key=lambda x: x[0], reverse=True)
        subgraph = [triple for score, triple in scored_triples[:self.n_triplets_per_sample]]

        subgraph_str = "\n".join(f"{item[0]} {item[1]} {item[2]}" for item in subgraph)
        query = TRIPLET_TO_SENTENCE_TEMPLATE.replace("##TRIPLET_PLACEHOLDER##", subgraph_str)
        _, summary, _, _usage = self.model.understand(query)

        return target_descriptions, parse_string(summary), usage + _usage

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        descriptions, retrieval, usage = self.retrieve(patient, candidates=candidates)
        medical_knowledge = "\n".join(descriptions) + "\n" + retrieval

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
        return {"query": prompt, "response": response, "prediction": prediction, "retrieval": retrieval, "usage": usage.dict()}
