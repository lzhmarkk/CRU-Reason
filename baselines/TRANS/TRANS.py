import os
import json
import torch_geometric as pyg
from baselines.TRANS.HGT import HGT
from baselines.TRANS.layers.TSEncoder import *
from baselines.TRANS.layers.Transformer import TransformerLayer
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
from typing import Dict
from baselines.TRANS.patient_graph import PatientGraph
from baselines.TRANS.static import feats_to_nodes, graph_meta


class TRANS(BaseModel):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            embedding_dim: int,
            config: Dict,
            **kwargs
    ):
        super(TRANS, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode
        )

        with open("./baselines/TRANS/hyperparam.json", "r", encoding='utf-8') as f:
            hyperparam = json.load(f)

        self.embedding_dim = embedding_dim

        self.feat_tokenizers = {}
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.label_tokenizer = self.get_label_tokenizer()
        output_size = self.get_output_size(self.label_tokenizer)
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            self.add_feature_transform_layer(feature_key, input_info)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(feature_size=embedding_dim,
                                                             dropout=hyperparam["dropout"])

        self.tim2vec = Time2Vec(8)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.graph_model = HGT(hidden_channels=self.embedding_dim,
                               out_channels=output_size,
                               num_heads=hyperparam["num_heads"],
                               num_layers=hyperparam["num_layers"],
                               metadata=graph_meta)

        self.pe = hyperparam["pe_dim"]
        self.spatial_encoder = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.spatial_encoder[feature_key] = nn.Linear(self.pe * 2, embedding_dim)  # .to(self.device)
        self.alpha = 0.8

        self.cache_path = os.path.join("./cache", f"{config['dataset']}-{config['task']}", "TRANS", config['exp'])
        os.makedirs(self.cache_path, exist_ok=True)
        self.graph_generator = PatientGraph(tokenizer=self.feat_tokenizers,
                                            dim=self.embedding_dim,
                                            device=self.device,
                                            cache_path=self.cache_path,
                                            trans_dim=self.pe,
                                            directed=False)

    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        return feature

    def process_seq(self, patients):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(patients[feature_key])
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.any(x != 0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits, patient_emb

    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for node_type, x in graph_list[i].x_dict.items():
                if node_type != 'visit':
                    if self.pe:
                        lpe = graph_list[i][node_type].laplacian_pe.to(self.device)
                        rws = graph_list[i][node_type].random_walk_se.to(self.device)
                        se = self.spatial_encoder[feats_to_nodes[node_type]](torch.cat([lpe, rws], dim=-1))
                        graph_list[i][node_type].x = f[feats_to_nodes[node_type]] + se
                    else:
                        graph_list[i][node_type].x = f[feats_to_nodes[node_type]]

                if node_type == 'visit':
                    timevec = self.tim2vec(torch.tensor(graph_list[i]['visit'].time, dtype=torch.float32, device=self.device))
                    num_visit = graph_list[i]['visit'].x.shape[0]
                    graph_list[i]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec], dim=-1)

        return pyg.data.Batch.from_data_list(graph_list).to(self.device)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patients, graphs = [], []
        for i in range(len(kwargs['conditions'])):
            patient_id = kwargs['patient_id'][i] + "_" + kwargs['visit_id'][i]

            patient = {}
            for feature_key in self.feature_keys:
                patient[feature_key] = kwargs[feature_key][i]

            graph = self.graph_generator.generate_graph_cached(patient_id, patient)

            patients.append(patient)
            graphs.append(graph)

        seq_logits, patient_emb = self.process_seq(kwargs)

        graph_data = self.process_graph_fea(graphs, patient_emb)

        logits = self.alpha * self.graph_model(graph_data.edge_index_dict, graph_data) + (1 - self.alpha) * seq_logits

        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        return results
