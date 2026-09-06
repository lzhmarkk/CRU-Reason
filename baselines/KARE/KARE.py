from typing import List, Dict
from pyhealth.datasets import SampleEHRDataset

from graph.graphloader import load_graph
from modules.post_process import POST_PROCESS_FN

from baselines.CoT.CoT import CoT
from baselines.CoT.prompt import TASK_DESCRIPTION, OUTPUT_INSTRUCTION, OUTPUT_FORMAT
from baselines.CoT.context.base_context import get_patient_embedding
from baselines.KARE.prompt import KARE_TEMPLATE
from baselines.KARE.dataset.context import KARE_context


class KARE(CoT):
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
        _, embeddings, _, _ = get_patient_embedding(kwargs['test_set'], task=self.task,
                                                    output_dir=f"./cache/{config['dataset']}-{config['task']}/CoT/test/")
        self.selected_summaries = KARE_context(sample_dataset=kwargs['test_set'],
                                               embeddings=embeddings, g=self.graph, model=self.model, task=self.task,
                                               output_dir=f"./cache/{config['dataset']}-{config['task']}/KARE/test/")

    def generate_reasoning(self, sample_id, patient, patient_context, similar_patient=None, candidates=None):
        medical_knowledge = self.selected_summaries[sample_id]
        medical_knowledge = [f"{id + 1}: {knowledge}" for id, knowledge in enumerate(medical_knowledge)]
        medical_knowledge = "\n".join(k.replace("\n", "") for k in medical_knowledge)

        if candidates is not None and self.candidate_fn is not None:
            patient_context += self.format_candidates(candidates)

        few_shots = self.format_fewshot_examples(similar_patient)

        prompt = KARE_TEMPLATE \
            .replace("##TASK_PLACEHOLDER##", TASK_DESCRIPTION[self.task]) \
            .replace("##CONTEXT_PLACEHOLDER##", patient_context) \
            .replace("##FEWSHOT_PLACEHOLDER##", few_shots) \
            .replace("##KNOWLEDGE_PLACEHOLDER##", medical_knowledge) \
            .replace("##INSTRUCTION_PLACEHOLDER##", OUTPUT_INSTRUCTION[self.task].replace("4. P", "5. P")) \
            .replace("##FORMAT_PLACEHOLDER##", OUTPUT_FORMAT[self.task].replace("4. P", "5. P"))  # update number 4->5

        _, (response, prediction), _, usage = self.model.understand(query=prompt, post_process=POST_PROCESS_FN[self.task])
        return {"query": prompt, "response": response, "prediction": prediction, "usage": usage.dict()}
