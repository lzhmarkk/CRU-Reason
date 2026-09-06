import torch
from typing import Dict
from pyhealth.models import StageNet


class MyStageNet(StageNet):
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        distance = []
        mask_dict = {}
        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 2: [[code1, code2], [code3, ...], ...]
            assert (dim_ == 3) and (type_ == str)
            x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key])
            # (patient, visit, event)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x != 0, dim=2)
            mask[:, 0] = 1  # !!fixed!!: ensure the first visit is not masked
            mask_dict[feature_key] = mask

            time = None
            if self.time_keys is not None:
                input_info = self.dataset.input_info[self.time_keys[idx]]
                dim_, type_ = input_info["dim"], input_info["type"]
                if (dim_ != 2) or (type_ not in [float, int]):
                    raise ValueError("Time interval must be 2-dim float or int.")
                time, _ = self.padding2d(kwargs[self.time_keys[idx]])
                time = torch.tensor(time, dtype=torch.float, device=self.device)
            x, _, cur_dis = self.stagenet[feature_key](x, time=time, mask=mask)
            patient_emb.append(x)
            distance.append(cur_dis)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)

        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
