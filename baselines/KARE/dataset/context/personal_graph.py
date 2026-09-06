import os
import json
from tqdm import tqdm
from graph.clustering.cluster import load_mapping
from graph.graphloader import load_graph, Graph


def filter_func(patient, visit_id, entity_mapping, mode):
    concepts = []
    for concept in patient[mode][visit_id]:
        if concept in entity_mapping:
            concepts.append(concept)
    return concepts


def read_indirect_nodes(main_concept, graph: Graph, entity_mapping):
    main_concept = entity_mapping[main_concept]
    subgraph = graph.sample(graph.entity2id[main_concept], num_hops=1)
    indirect_nodes = set(graph.id2entity[e.item()] for e in subgraph.entity)
    return indirect_nodes


def patient_graph_construct(task, patient, graph, entity_mapping):
    num_visit = len(patient['conditions'])
    direct_nodes, indirect_nodes = set(), set()

    for i in range(num_visit):
        feature_keys = ['conditions', 'procedures'] if task in ['drugrec']\
            else ['conditions', 'procedures', 'drugs']
        for feature in feature_keys:
            concepts = filter_func(patient, i, entity_mapping, feature)

            for concept in concepts:
                direct_node = entity_mapping[concept]
                direct_nodes.add(direct_node)
                indirect_nodes_condition = read_indirect_nodes(concept, graph, entity_mapping)
                indirect_nodes.update(indirect_nodes_condition)

    return direct_nodes, indirect_nodes


def get_patient_graph(task, sample_dataset, graph, output_dir):
    entity_mapping, _, _, _ = load_mapping()

    print("Constructing patient graphs ...")
    output_dir = os.path.join(output_dir, "patient_graph")
    os.makedirs(output_dir, exist_ok=True)

    patient_nodes = {}
    for idx, patient in tqdm(enumerate(sample_dataset), total=len(sample_dataset)):
        patient_id = f"{patient['patient_id']}_{patient['visit_id']}"
        output_file = os.path.join(output_dir, f"{patient_id}.json")

        if os.path.exists(output_file):
            with (open(output_file, 'r', encoding='utf-8') as f):
                patient_node = json.load(f)

        else:
            # node id of cluster graph
            direct_nodes, indirect_nodes = patient_graph_construct(task, patient, graph, entity_mapping)
            patient_node = {"direct": list(direct_nodes),
                            "indirect": list(indirect_nodes)}

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(patient_node, f)

        patient_nodes[patient_id] = patient_node

    return patient_nodes
