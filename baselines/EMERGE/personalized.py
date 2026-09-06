from graph.graphloader import Graph


def personalized_graph(patient, graph: Graph):
    n_visits = len(patient['conditions'])

    visit_concepts = []  # concept of each visit

    for visit in range(n_visits):
        for concept_key in patient.keys():
            for concept in patient[concept_key][visit]:
                if concept in graph.entity2id:
                    concept = graph.entity2id[concept]
                    visit_concepts.append(concept)

    concepts = list(set(visit_concepts))
    subgraph = graph.sample(concepts, num_hops=1)

    concepts = [graph.id2entity[c] for c in concepts]
    triples = [(graph.id2entity[src.item()], graph.id2relation[rel.item()], graph.id2entity[dst.item()]) for (src, dst), rel in zip(subgraph.edge_index.T, subgraph.relation)]
    return concepts, triples
