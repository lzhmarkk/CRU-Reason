import os
import torch
import pickle
import networkx as nx
import torch_geometric as pyg
from torch_geometric.utils import k_hop_subgraph


def read_graph_from_file(path):
    def strip(string):
        if string.startswith('\'') and string.endswith('\''):
            string = string[1:-1]
        elif string.startswith('\"') and string.endswith('\"'):
            string = string[1:-1]
        string = string.strip()
        return string

    relations = set()
    skipped = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip().split('\t')) != 3:
                skipped += 1
                continue
            head, relation, tail = map(strip, line.strip().split('\t'))
            relations.add((head, relation, tail))

    print(f"Knowledge Graph {path} loaded. Total {len(relations)} edges. Skipped {skipped} lines.")

    return relations


def load_embedding():
    with open(f"./graph/embedding/entity_embeddings.pkl", 'rb') as f:
        entity_embeddings = pickle.load(f)

    with open(f"./graph/embedding/relation_embeddings.pkl", 'rb') as f:
        relation_embeddings = pickle.load(f)

    return entity_embeddings, relation_embeddings


def load_cluster_embedding():
    with open(f"./graph/clustering/cluster_entity_embeddings.pkl", 'rb') as f:
        entity_embeddings = pickle.load(f)
    with open(f"./graph/clustering/cluster_relation_embeddings.pkl", 'rb') as f:
        relation_embeddings = pickle.load(f)

    return entity_embeddings, relation_embeddings


class Graph:
    def __init__(self, edges, entity_embeddings, relation_embeddings, directed=False):
        self.entities = sorted(list(set([e[0] for e in edges] + [e[2] for e in edges])))
        self.relations = sorted(list(set(e[1] for e in edges)))
        self.entity2id, self.id2entity = self.construct_mapping(self.entities)
        self.relation2id, self.id2relation = self.construct_mapping(self.relations)

        self.n_edges = len(edges)
        self.n_nodes = len(self.entities)
        self.n_relations = len(self.relations)

        self.entity2embedding = {k: torch.tensor(emb) for k, emb in entity_embeddings.items()}
        self.relation2embedding = {k: torch.tensor(emb) for k, emb in relation_embeddings.items()}
        self.entity_embedding = torch.tensor([entity_embeddings[self.id2entity[i]] for i in range(self.n_nodes)])
        self.relation_embedding = torch.tensor([relation_embeddings[self.id2relation[i]] for i in range(self.n_relations)])

        self.graph = self.to_pyg(edges, directed=directed)

    def construct_mapping(self, elements):
        element2id, id2element = {}, {}
        for id, element in enumerate(elements):
            element2id[element] = id
            id2element[id] = element
        return element2id, id2element

    def to_pyg(self, edges, directed):
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        nodes = list(self.id2entity.keys())
        for n in nodes:
            G.add_node(n, entity=n)

        for edge in edges:
            u, r, v = edge
            u = self.entity2id[u]
            r = self.relation2id[r]
            v = self.entity2id[v]
            G.add_edge(u, v, relation=r)

        G = pyg.utils.from_networkx(G)  # this operation is slow
        return G

    def sample(self, nodes, num_hops):
        nodes, _, _, edge_mask = k_hop_subgraph(nodes, num_hops=num_hops, edge_index=self.graph.edge_index)
        subgraph = self.graph.edge_subgraph(edge_mask)
        subgraph = subgraph.subgraph(nodes)  # relabel nodes
        assert subgraph.edge_index.shape[1] == subgraph.relation.shape[0]
        return subgraph


def load_graph(cluster=False, directed=False, use_cache=True):
    if directed:
        file = "./graph/kg_cluster_di.pkl" if cluster else "./graph/kg_raw_di.pkl"
    else:
        file = "./graph/kg_cluster.pkl" if cluster else "./graph/kg_raw.pkl"

    if use_cache and os.path.exists(file):
        with open(file, 'rb') as f:
            graph = pickle.load(f)

    else:
        if cluster:
            edges = read_graph_from_file("./graph/kg_cluster.txt")
            entity_embeddings, relation_embeddings = load_cluster_embedding()
        else:
            edges = read_graph_from_file("./graph/kg_raw.txt")
            entity_embeddings, relation_embeddings = load_embedding()

        graph = Graph(edges, entity_embeddings, relation_embeddings, directed=directed)
        if use_cache:
            with open(file, 'wb') as f:
                pickle.dump(graph, f)

    assert isinstance(graph, Graph)
    return graph
