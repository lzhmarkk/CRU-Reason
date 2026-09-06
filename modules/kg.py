import torch
from collections import defaultdict
from graph.graphloader import Graph, load_graph
from torch_geometric.utils import contains_self_loops, remove_self_loops

graph: Graph = load_graph(cluster=False, directed=True)
edge_index = graph.graph.edge_index
print("edge_index.shape", edge_index.shape)

if contains_self_loops(edge_index):
    edge_index, edge_attr = remove_self_loops(edge_index, graph.graph.relation)
    graph.graph.edge_index = edge_index
    graph.graph.relation = edge_attr
    print("Self-loop detected, Removed")

adjacency_list, adjacency_list_inv = defaultdict(list), defaultdict(list)
for eid, (src, dst) in enumerate(edge_index.t().tolist()):
    adjacency_list[src].append((eid, dst))
    adjacency_list_inv[dst].append((eid, src))


def get_triplet(src, dst):
    for edge_ids, tail in adjacency_list[src]:
        if tail == dst:
            edge_type = graph.graph.relation[edge_ids].item()
            edge_name = graph.id2relation[edge_type]

            src_name = graph.id2entity[src]
            dst_name = graph.id2entity[dst]
            return src_name, edge_name, dst_name

    return None, None, None


def get_neighbors(concept, candidate_ids: list, direction='both'):
    if concept not in graph.entity2id or len(candidate_ids) <= 0: return []

    if isinstance(candidate_ids[0], str):
        candidate_ids = [graph.entity2id[candidate] for candidate in candidate_ids]

    triplets = []
    concept_id = graph.entity2id[concept]
    if direction == 'forward' or direction == 'both':
        target_triplets = [(concept_id, graph.graph.relation[eid].item(), next)
                           for eid, next in adjacency_list[concept_id]
                           if next in candidate_ids]
        triplets += target_triplets

    if direction == 'backward' or direction == 'both':
        target_triplets_inv = [(prev, graph.graph.relation[eid].item(), concept_id)
                               for eid, prev in adjacency_list_inv[concept_id]
                               if prev in candidate_ids]
        triplets += target_triplets_inv

    triplet_str = [(graph.id2entity[src], graph.id2relation[rel], graph.id2entity[dst]) for src, rel, dst in triplets]
    return triplet_str
