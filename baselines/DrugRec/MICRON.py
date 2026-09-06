import os
import copy
import torch
import numpy as np
from pyhealth.models import MICRON, MICRONLayer
from pyhealth.models.utils import get_last_visit
from pyhealth.datasets import SampleEHRDataset
from typing import List, Dict, Optional, Tuple
from data.dataloader import load_mappings
from pyhealth import BASE_CACHE_PATH as CACHE_PATH


class MyMICRONLayer(MICRONLayer):
    def forward(
            self,
            patient_emb: torch.tensor,
            drugs: torch.tensor,
            mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:

        if mask is None:
            mask = torch.ones_like(patient_emb[:, :, 0])

        # (patient, visit, hidden_size)
        health_rep = self.health_net(patient_emb)
        drug_rep = self.prescription_net(health_rep)
        logits = self.fc(drug_rep)
        logits_last_visit = get_last_visit(logits, mask)
        bce_loss = self.bce_loss_fn(logits_last_visit, drugs)

        # fix: visit len = 1 lead to nan loss
        if health_rep.shape[1] > 1:
            # (batch, visit-1, input_size)
            health_rep_last = health_rep[:, :-1, :]
            # (batch, visit-1, input_size)
            health_rep_cur = health_rep[:, 1:, :]
            # (batch, visit-1, input_size)
            health_rep_residual = health_rep_cur - health_rep_last
            drug_rep_residual = self.prescription_net(health_rep_residual)
            logits_residual = self.fc(drug_rep_residual)
            rec_loss = self.compute_reconstruction_loss(logits, logits_residual, mask)
        else:
            rec_loss = 0

        loss = bce_loss + self.lam * rec_loss
        y_prob = torch.sigmoid(logits_last_visit)

        return loss, y_prob


class MyMICRON(MICRON):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            **kwargs
    ):
        # fix: convert sample['drugs'] to ATC format
        _, _, drug_dict = load_mappings()
        self.drug_dict_inv = {name: code for code, name in drug_dict.items()}
        dataset = copy.deepcopy(dataset)
        for sample in dataset:
            sample['drugs'] = [self.drug_dict_inv[d] for d in sample['drugs']]

        kwargs.pop("mode")
        kwargs.pop("feature_keys")
        kwargs.pop("label_key")

        super(MICRON, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # fix: use customized Layer
        self.micron = MyMICRONLayer(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            num_drugs=self.label_tokenizer.get_vocabulary_size(),
            **kwargs
        )

        # save ddi adj
        ddi_adj = self.generate_ddi_adj()
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj)

    def forward(self,
                conditions: List[List[List[str]]],
                procedures: List[List[List[str]]],
                drugs: List[List[str]],
                **kwargs
                ) -> Dict[str, torch.Tensor]:
        # fix: convert 'drugs' to ATC format
        drugs = [[self.drug_dict_inv[d] for d in _drugs] for _drugs in drugs]

        return super().forward(conditions, procedures, drugs, **kwargs)
