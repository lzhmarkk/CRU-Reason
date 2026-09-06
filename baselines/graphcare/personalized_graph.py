import torch
import itertools
from graph.clustering.cluster import load_mapping
from graph.graphloader import Graph


def personalized_graph(patient, graph: Graph, feature_key, device):
    ett2cluster, _, rel2cluster, _ = load_mapping()

    n_visits = len(patient['conditions'])

    visit_concepts = []  # concept of each visit
    for visit in range(n_visits):
        _visit_concepts = set()
        for concept_key in feature_key:
            for concept in patient[concept_key][visit]:
                if concept in ett2cluster:
                    cluster = ett2cluster[concept]
                    concept = graph.entity2id[cluster]
                    _visit_concepts.add(concept)
        visit_concepts.append(list(_visit_concepts))

    all_entities = list(set(itertools.chain(*visit_concepts)))
    all_entities = torch.LongTensor(all_entities).to(device)
    graph.graph = graph.graph.to(device)
    graph = graph.sample(all_entities, num_hops=1)

    graph.all_entities = all_entities
    graph.visit_concepts = visit_concepts

    return graph
