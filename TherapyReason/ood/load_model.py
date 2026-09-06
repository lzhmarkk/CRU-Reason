import copy
import torch
import random
from typing import Dict

from pyhealth.datasets import SampleEHRDataset
from pyhealth.tokenizer import Tokenizer
from pyhealth.models import BaseModel

from data.dataloader import load_mappings
from utils.load_model import get_model


class OODWrapper:
    def __init__(self,
                 model: BaseModel,
                 source_dataset: SampleEHRDataset,
                 target_dataset: SampleEHRDataset):

        self.model = model
        self.mode = self.model.mode

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        target_feature_tokenizers = {}
        for feature_key in model.feature_keys:
            target_feature_tokenizers[feature_key] = Tokenizer(
                tokens=self.target_dataset.get_all_tokens(key=feature_key),
                special_tokens=["<pad>", "<unk>"]
            )
        target_label_tokenizer = Tokenizer(
            self.target_dataset.get_all_tokens(key=model.label_key),
            special_tokens=[]
        )

        self.source_feat_tokenizers = model.feat_tokenizers if hasattr(model, 'feat_tokenizers') else model.get_feature_tokenizers()
        if self.model.__class__.__name__ in ['MyMICRON', 'MyGAMENet', 'MyMoleRec', 'MySafeDrug']:  # models that internally use ATC codes
            _, _, drug_dict = load_mappings()
            atc_vocabulary = model.label_tokenizer.vocabulary
            new_idx2token = {}
            new_token2idx = {}
            for idx, token in atc_vocabulary.idx2token.items():
                token = drug_dict[token]
                new_idx2token[idx] = token
                new_token2idx[token] = idx
            self.source_label_tokenizer = copy.deepcopy(model.label_tokenizer)
            self.source_label_tokenizer.vocabulary.idx2token = new_idx2token
            self.source_label_tokenizer.vocabulary.token2idx = new_token2idx
        else:
            self.source_label_tokenizer = model.label_tokenizer

        self.target_feat_tokenizers = target_feature_tokenizers
        self.target_label_tokenizer = target_label_tokenizer

    def to(self, device):
        self.model.to(device)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def eval(self):
        self.model.eval()

    def __call__(self, **kwargs) -> Dict[str, torch.Tensor]:
        backup_label = copy.deepcopy(kwargs[self.model.label_key])

        # input adaption: removing concepts from target_dataset (input) that are not in source_dataset (model param)
        for feature_key in self.model.feature_keys:
            mapped_feature_value = []
            for patient in kwargs[feature_key]:
                mapped_patient = []
                for visit in patient:
                    mapped_visit = [concept for concept in visit if concept in self.source_feat_tokenizers[feature_key].vocabulary]
                    if len(mapped_visit) == 0:
                        mapped_visit = [random.choice(list(self.target_feat_tokenizers[feature_key].vocabulary.token2idx))]

                    mapped_patient.append(mapped_visit)
                mapped_feature_value.append(mapped_patient)
            kwargs[feature_key] = mapped_feature_value

        kwargs[self.model.label_key] = [[label for label in patient if label in self.source_label_tokenizer.vocabulary]
                                        for patient in kwargs[self.model.label_key]]

        result = self.model(**kwargs)

        batch_size = result['y_prob'].shape[0]
        target_y_prob = torch.zeros(batch_size, len(self.target_label_tokenizer.vocabulary), device=self.model.device)

        # map output from source_dataset vocabulary to target_dataset vocabulary
        for i, token in self.source_label_tokenizer.vocabulary.idx2token.items():
            if token in self.target_label_tokenizer.vocabulary:
                j = self.target_label_tokenizer.vocabulary(token)
                target_y_prob[:, j] = result['y_prob'][:, i]

        # label adaption
        target_y_true = self.model.prepare_labels(backup_label, self.target_label_tokenizer)

        return {"loss": result["loss"], "y_prob": target_y_prob, "y_true": target_y_true}


def get_ood_model(config, mode, feature_keys, source_dataset, target_dataset):
    model = get_model(config, mode, feature_keys, full_dataset=source_dataset)
    model = OODWrapper(model, source_dataset, target_dataset)
    return model
