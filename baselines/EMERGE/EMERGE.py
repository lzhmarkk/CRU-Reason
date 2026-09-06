import os
import json
import torch
import random
import itertools
import torch.nn as nn
from pathlib import Path
from typing import List, Dict

from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset

from utils.llm import load_model
from utils.multiprocess import multiprocess_call
from graph.graphloader import load_graph
from graph.embedding.embedding import generate_embeddings
from baselines.EMERGE.personalized import personalized_graph


class EMERGE(BaseModel):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            config: Dict,
            **kwargs
    ):
        super(EMERGE, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode
        )

        self.embedding_dim = int(kwargs['embedding_dim'])
        self.label_tokenizer = self.get_label_tokenizer()
        self.task = config['task']

        self.kg = load_graph(cluster=False, use_cache=True)

        self.model = load_model(config)
        self.embedding_model = "text-embedding-3-large"
        self.n_workers = 8

        self.cache_path = os.path.join("./cache", f"{config['dataset']}-{self.task}", "EMERGE", config['exp'])
        os.makedirs(self.cache_path, exist_ok=True)

        n_trainset_size, n_valset_size = 8000, 500  # since LLMs are used, limit the cost
        self.summary_embedding, self.note_embedding = self.generate_summary_embedding(train_set=kwargs["train_set"][:n_trainset_size],
                                                                                      val_set=kwargs["val_set"][:n_valset_size],
                                                                                      test_set=kwargs["test_set"])

        # self.note_embedding_proj = nn.Linear(3072, self.embedding_dim)
        # self.summary_embedding_proj = nn.Linear(3072, self.embedding_dim)
        self.note_embedding_proj = nn.Identity()
        self.summary_embedding_proj = nn.Identity()
        self.concat_text_fusion_proj = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        output_size = self.get_output_size(self.label_tokenizer)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(p=0.25),
            nn.Linear(self.embedding_dim, output_size)
        )

    def generate_summary_embedding(self, train_set, val_set, test_set):
        def process_single(key, sample):
            patient_id = sample['patient_id'] + "_" + sample['visit_id']
            patient = {feature_key: sample[feature_key] for feature_key in self.feature_keys}

            cache_file = Path(os.path.join(self.cache_path, f"{patient_id}.json"))
            if cache_file.exists():
                cache = json.loads(cache_file.read_text(encoding='utf-8'))
                summary_embedding, note_embedding = cache["summary_embedding"], cache['note_embedding']

            else:
                summary, summary_embedding, note, note_embedding, usage = self.get_embedding(patient)
                cache = {"summary": summary, "summary_embedding": summary_embedding,
                         "note": note, "note_embedding": note_embedding,
                         "usage": usage.dict()}
                cache_file.write_text(json.dumps(cache, indent=4), encoding='utf-8')

            return key, (patient_id, summary_embedding, note_embedding)

        # embedding
        summary_embedding_patients, note_embedding_patients = {}, {}

        for dataset in [train_set, val_set, test_set]:
            print("Generating summary embedding for dataset")
            results = multiprocess_call(process_single, dataset, n_workers=self.n_workers)
            for patient_id, summary_embedding, note_embedding in results:
                summary_embedding_patients[patient_id] = summary_embedding
                note_embedding_patients[patient_id] = note_embedding

        return summary_embedding_patients, note_embedding_patients

    def get_note(self, patient):
        note = ""
        for feature_key in patient.keys():
            values = ", ".join(itertools.chain(*patient[feature_key]))
            note += f"{feature_key}: {values}\n"

        return note

    def get_prompt(self, entities, triples):
        entity_str = "[" + ", ".join(entities) + "]"
        # Small patient-specific subgraphs may not contain five triples per
        # entity.  Sampling more items than are available raises ValueError.
        sel_triples = random.sample(triples, min(len(triples), len(entities) * 5))
        triple_str = "\n".join(f"({t[0]}, {t[1]}, {t[2]})" for t in sel_triples)

        if self.task == 'mortality':
            task_desc = "in-hospital mortality prediction"
        elif self.task == 'readmission':
            task_desc = "15-day readmission prediction"
        elif self.task == 'lenofstay':
            task_desc = "in-hospital stay length"
        elif self.task == 'drugrec':
            task_desc = "medication recommendation"
        elif self.task == 'diagnosis':
            task_desc = "diagnosis prediction"
        else:
            raise ValueError()

        return f"""\
As an experienced clinical professor, you have been provided with the following information to assist in summarizing a patient's health status:
- Potential abnormal features exhibited by the patient
- Possible diseases the patient may be suffering from
- Definitions and descriptions of the corresponding diseases
- Knowledge graph triples specific to these diseases
Using this information, please create a concise and clear summary of the patient's health status. \
Your summary should be informative and beneficial for various healthcare prediction tasks (e.g. {task_desc}).
Please provide your summary directly without any additional explanations.

# Potential diseases
{entity_str}

# Disease relationships
{triple_str}
"""

    def get_embedding(self, patient):
        # extract
        concepts, triples = personalized_graph(patient, self.kg)

        prompts = self.get_prompt(concepts, triples)
        _, summary, _, usage = self.model.understand(query=prompts)
        summary_embedding = generate_embeddings([summary], model=self.embedding_model, dimension=self.embedding_dim)
        summary_embedding = list(summary_embedding.values())[0]

        note = self.get_note(patient)
        note_embedding = generate_embeddings([note], model=self.embedding_model, dimension=self.embedding_dim)
        note_embedding = list(note_embedding.values())[0]

        return summary, summary_embedding, note, note_embedding, usage

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # kwargs: {'visit_id': List[str], 'patient_id': List[str], 'conditions': List[List[List[str]]]], 'label': List[int]}
        summary_embedding_patients, note_embedding_patients = [], []
        for i in range(len(kwargs['conditions'])):
            patient_id = kwargs['patient_id'][i] + "_" + kwargs['visit_id'][i]
            summary_embedding_patients.append(self.summary_embedding[patient_id])
            note_embedding_patients.append(self.note_embedding[patient_id])

        # fusion
        h_rag = torch.tensor(summary_embedding_patients, device=self.device)
        h_note = torch.tensor(note_embedding_patients, device=self.device)
        h_rag = self.summary_embedding_proj(h_rag)
        h_note = self.note_embedding_proj(h_note)
        h_text = torch.cat([h_rag, h_note], dim=-1)
        h_text = self.concat_text_fusion_proj(h_text)

        # output
        patient_emb = h_text
        # (patient, label_size)
        logits = self.mlp(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        return results
