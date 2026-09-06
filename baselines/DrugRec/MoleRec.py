import copy
import torch
from pyhealth.models import MoleRec
from pyhealth.datasets import SampleEHRDataset
from typing import List, Dict

from data.dataloader import load_mappings


class MyMoleRec(MoleRec):
    def __init__(self,
                 dataset: SampleEHRDataset,
                 embedding_dim: int = 64,
                 hidden_dim: int = 64,
                 num_rnn_layers: int = 1,
                 num_gnn_layers: int = 4,
                 dropout: float = 0.5,
                 **kwargs,
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

        super().__init__(dataset, embedding_dim, hidden_dim, num_rnn_layers, num_gnn_layers, dropout, **kwargs)

    def forward(self,
                conditions: List[List[List[str]]],
                procedures: List[List[List[str]]],
                drugs: List[List[str]],
                **kwargs,
                ) -> Dict[str, torch.Tensor]:
        # fix: convert 'drugs' to ATC format
        drugs = [[self.drug_dict_inv[d] for d in _drugs] for _drugs in drugs]

        return super().forward(conditions, procedures, drugs, **kwargs)
