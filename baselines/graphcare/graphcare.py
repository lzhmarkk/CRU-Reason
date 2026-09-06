import json
import os
import torch
import torch.nn as nn
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
from typing import List, Dict
from .BAT import BiAttentionGNNConv
from graph.graphloader import load_graph
from .personalized_graph import personalized_graph
import torch_geometric as pyg


class GraphCare(BaseModel):
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            config: Dict,
            **kwargs
    ):
        super(GraphCare, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode
        )

        self.kg = load_graph(cluster=True, use_cache=True)

        with open("./baselines/graphcare/hyperparam.json", "r", encoding='utf-8') as f:
            hyperparam = json.load(f)

        self.num_nodes: int = self.kg.n_nodes
        self.num_relations: int = self.kg.n_relations
        self.max_visits: int = kwargs['max_visits']
        self.embedding_dim: int = hyperparam['embed_dim']  # emb dim of word embedding
        self.hidden_dim: int = kwargs['embedding_dim']  # emb dim for modules
        self.num_layers: int = hyperparam['num_layers']
        self.dropout: float = hyperparam['dropout']
        self.decay_rate: float = hyperparam['decay_rate']
        self.patient_mode: str = hyperparam['patient_mode']
        self.use_edge_attn: bool = hyperparam['use_edge_attn']

        #
        j = torch.arange(self.max_visits).float()
        self.lambda_j = torch.exp(self.decay_rate * (self.max_visits - j)).unsqueeze(0).reshape(1, self.max_visits, 1).float()

        if not hyperparam['use_node_emb']:
            self.node_emb = nn.Embedding(self.num_nodes, self.embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(self.kg.entity_embedding, freeze=hyperparam['freeze'])
        if not hyperparam['use_relation_emb']:
            self.rel_emb = nn.Embedding(self.num_relations, self.embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(self.kg.relation_embedding, freeze=hyperparam['freeze'])

        self.lin = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()

        self.use_alpha: bool = hyperparam['use_alpha']
        self.use_beta: bool = hyperparam['use_beta']
        self.use_attn_init: bool = hyperparam['use_attn_init']
        if self.use_attn_init:
            attn_weights = torch.tensor(1)
        else:
            attn_weights = None

        for layer in range(1, self.num_layers + 1):
            if self.use_alpha:
                self.alpha_attn[str(layer)] = nn.ModuleList([  # factored to save memory
                    nn.Embedding(self.num_nodes, self.embedding_dim // 2),
                    nn.Embedding(self.num_nodes, self.embedding_dim // 2),
                    nn.Embedding(1, self.num_nodes)
                ])

                if self.use_attn_init:
                    attn_init_matrix = torch.eye(self.num_nodes).float() * attn_weights.float()  # Multiply the identity matrix by attn_init
                    self.alpha_attn[str(layer)].weight.data.copy_(attn_init_matrix)  # Copy the modified attn_init_matrix to the weights
                else:
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)][0].weight)
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)][1].weight)

            if self.use_beta:
                self.beta_attn[str(layer)] = nn.Linear(self.num_nodes, 1)
                nn.init.xavier_normal_(self.beta_attn[str(layer)].weight)

            self.conv[str(layer)] = BiAttentionGNNConv(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                       edge_dim=self.hidden_dim, edge_attn=self.use_edge_attn, eps=0.)

        self.label_tokenizer = self.get_label_tokenizer()
        output_size = self.get_output_size(self.label_tokenizer)
        self.MLP = nn.Linear(self.hidden_dim * (2 if self.patient_mode == 'joint' else 1), output_size)

        self.cache_path = os.path.join("./cache", f"{config['dataset']}-{config['task']}", "GraphCare", config['exp'])
        os.makedirs(self.cache_path, exist_ok=True)

    def get_personalized_graph(self, patient_id, patient):
        cache_path = os.path.join(self.cache_path, f"{patient_id}.pt")
        if os.path.exists(cache_path):
            graph = torch.load(cache_path, map_location=self.device)
        else:
            graph = personalized_graph(patient, self.kg, feature_key=self.feature_keys, device=self.device)
            torch.save(graph, cache_path)

        visit_node_padded = torch.zeros(len(graph.visit_concepts), self.kg.n_nodes, dtype=torch.bool)
        for visit, concepts in enumerate(graph.visit_concepts):
            visit_node_padded[visit][concepts] = True
        visit_node_padded = torch.nn.functional.pad(visit_node_padded, (0, 0, 0, self.max_visits - len(visit_node_padded)))
        graph.visit_node_padded = visit_node_padded.unsqueeze(0).to(self.device)  # (1, T, N)
        graph.visit_node = graph.visit_node_padded.any(dim=1)  # (1, N)
        return graph

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # kwargs: {'visit_id': List[str], 'patient_id': List[str], 'conditions': List[List[List[str]]]], 'label': List[int]}

        # construct input graph
        personal_graph = []
        for i in range(len(kwargs['conditions'])):
            patient_id = kwargs['patient_id'][i] + "_" + kwargs['visit_id'][i]
            patient = {}
            for feature_key in self.feature_keys:
                patient[feature_key] = kwargs[feature_key][i]

            graph = self.get_personalized_graph(patient_id, patient)
            personal_graph.append(graph)

        batched_graph = pyg.data.Batch.from_data_list(personal_graph).to(self.device)
        edge_index = batched_graph.edge_index  # (E,)
        batch_size = batched_graph.batch_size
        batch = batched_graph.batch  # (BN) value refers to the batch index for each node
        visit_node_padded = batched_graph.visit_node_padded  # (B, T, N) binary
        visit_node = batched_graph.visit_node  # (B, N) binary

        # features
        x = self.node_emb(batched_graph.entity)
        x = self.lin(x)
        edge_attr = self.rel_emb(batched_graph.relation)
        edge_attr = self.lin(edge_attr)

        # modules
        for layer in range(1, self.num_layers + 1):
            if self.use_alpha:
                alpha_weight = self.alpha_attn[str(layer)][0].weight @ self.alpha_attn[str(layer)][1].weight.T  # (N, N)
                alpha_bias = self.alpha_attn[str(layer)][2].weight
                alpha = visit_node_padded.float() @ alpha_weight + alpha_bias  # (B, T, N)
                alpha = torch.softmax(alpha, dim=1)  # (batch, max_visit, num_nodes)
            else:
                alpha = torch.ones((batch_size, self.max_visit, self.num_nodes)).to(self.device)

            if self.use_beta:
                beta = torch.tanh(self.beta_attn[str(layer)](visit_node_padded.float())) * self.lambda_j.to(self.device)
            else:
                beta = torch.ones((batch_size, self.max_visit, 1)).to(self.device)

            attn = alpha * beta  # (B, T, N)
            attn = torch.sum(attn, dim=1)  # (B, N)
            src_batch_id = batch[edge_index[0]]
            src_node_id = batched_graph.entity[edge_index[0]]
            attn = attn[src_batch_id, src_node_id].reshape(-1, 1)  # (E, 1)

            x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn)
            x = torch.relu(x)
            x = torch.dropout(x, p=0.5, train=self.training)

        # output
        all_x = []
        if self.patient_mode == "joint" or self.patient_mode == "graph":
            # patient graph embedding through global mean pooling
            x_graph = pyg.nn.global_mean_pool(x, batch)
            x_graph = torch.dropout(x_graph, p=self.dropout, train=self.training)
            all_x.append(x_graph)

        if self.patient_mode == "joint" or self.patient_mode == "node":
            # patient node embedding through local (direct EHR) mean pooling
            x_node = torch.stack([visit_node[i].view(1, -1).float() @ self.node_emb.weight / torch.sum(visit_node[i]) for i in range(batch_size)])
            x_node = self.lin(x_node).squeeze(1)
            x_node = torch.dropout(x_node, p=self.dropout, train=self.training)
            all_x.append(x_node)

        if len(all_x) == 2:
            x = torch.cat(all_x, dim=1)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        else:
            x = all_x[0]

        logits = self.MLP(x)

        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        return results
