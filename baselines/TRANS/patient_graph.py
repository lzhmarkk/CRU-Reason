import os
import pickle
import torch
from torch_geometric.data import HeteroData
from baselines.TRANS.layers.TSEncoder import AddGlobalLaplacianPE, AddMetaPathRandomWalkSE
from baselines.TRANS.static import feats_to_nodes


class PatientGraph:
    def __init__(self, tokenizer, dim, device, cache_path, trans_dim=0, directed=False):
        self.tokenizers = tokenizer
        self.di_edge = directed
        self.dim = dim
        self.se = False
        if trans_dim != 0:
            self.se = True
            self.global_pe_transform = AddGlobalLaplacianPE(k=trans_dim, device=device)
            self.local_se_transform = AddMetaPathRandomWalkSE(trans_dim, device=device)

        self.cache_path = cache_path

    def generate_graph_cached(self, patient_id, patient):
        cache_file = os.path.join(self.cache_path, f"{patient_id}.pt")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                graph = pickle.load(f)

        else:
            graph = self.generate_graph(patient)
            with open(cache_file, "wb") as f:
                pickle.dump(graph, f)

        return graph

    def generate_graph(self, sample):
        graph = HeteroData()
        num_visit = len(sample['procedures'])
        graph['visit'].x = torch.zeros(num_visit, self.dim)  # self.visits[dp].unsqueeze(0).expand(num_visit, -1) #torch.zeros(num_visit, self.dim)

        for feat in self.tokenizers:
            ntype = feats_to_nodes[feat]
            dp = self.tokenizers[feat].batch_encode_2d(sample[feat], padding=False)
            graph[ntype].x = torch.zeros(self.tokenizers[feat].get_vocabulary_size(), self.dim)
            iv = torch.tensor([[item for sublist in dp for item in sublist],
                               [index for index, sublist in enumerate(dp) for _ in sublist]], dtype=torch.int64)
            graph[ntype, 'in', 'visit'].edge_index = iv
            fiv = torch.flip(iv, [0])
            graph[ntype, 'in', 'visit'].edge_time = torch.tensor([index for index, sublist in enumerate(dp) for _ in sublist], dtype=torch.float32)
            graph['visit', 'has', ntype].edge_index = fiv

        graph['visit'].time = torch.arange(num_visit) * 1440  # no timestamp available for fairness, so assume one visit a day
        viv = torch.tensor([[i for i in range(num_visit - 1)], [i + 1 for i in range(num_visit - 1)]], dtype=torch.int64)
        graph['visit', 'connect', 'visit'].edge_index = viv

        if self.di_edge:
            graph['visit', 'connect', 'visit'].edge_index = viv
        else:
            graph['visit', 'connect', 'visit'].edge_index = torch.cat([viv, torch.flip(viv, [0])], dim=1)

        if self.se:
            graph = self.global_pe_transform.apply_laplacian_pe(graph)
            f_metapaths = [
                [('co', 'in', 'visit'), ('visit', 'has', 'pr'), ("pr", 'in', "visit"), ('visit', 'has', 'co')],
                [("pr", 'in', "visit"), ('visit', 'has', 'co'), ('co', 'in', 'visit'), ('visit', 'has', 'pr')],
            ]
            if 'drugs' in self.tokenizers:
                f_metapaths.append([("dh", 'in', "visit"), ('visit', 'has', 'dh')])
            graph = self.local_se_transform.forward(data=graph, metapaths=f_metapaths)
            for node_type in graph.node_types:
                if node_type not in ['co', 'pr', 'dh']:
                    del graph[node_type].laplacian_pe
                    del graph[node_type].random_walk_se

        return graph
