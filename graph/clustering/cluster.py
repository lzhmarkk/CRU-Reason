import os
import json
import pickle
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor, as_completed
from graph.graphloader import read_graph_from_file, load_embedding

ENT_THRESHOLD = 0.14
REL_THRESHOLD = 0.14
output_dir = "./graph/clustering/"


def load_mapping():
    with open(f"{output_dir}/original_to_new_entity_mapping.json", 'r', encoding='utf-8') as f:
        original_to_new_entity_mapping = json.load(f)
    with open(f"{output_dir}/new_to_original_entity_mapping.json", 'r', encoding='utf-8') as f:
        new_to_original_entity_mapping = json.load(f)

    with open(f"{output_dir}/original_to_new_relation_mapping.json", 'r', encoding='utf-8') as f:
        original_to_new_relation_mapping = json.load(f)
    with open(f"{output_dir}/new_to_original_relation_mapping.json", 'r', encoding='utf-8') as f:
        new_to_original_relation_mapping = json.load(f)

    return original_to_new_entity_mapping, new_to_original_entity_mapping, original_to_new_relation_mapping, new_to_original_relation_mapping


def save_mapping(original_to_new_entity_mapping, new_to_original_entity_mapping, original_to_new_relation_mapping, new_to_original_relation_mapping):
    print("Saving entity mappings...")
    with open(f"{output_dir}/original_to_new_entity_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(original_to_new_entity_mapping, f, indent=4)
    with open(f"{output_dir}/new_to_original_entity_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(new_to_original_entity_mapping, f, indent=4)

    print("Saving relation mappings...")
    with open(f"{output_dir}/original_to_new_relation_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(original_to_new_relation_mapping, f, indent=4)
    with open(f"{output_dir}/new_to_original_relation_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(new_to_original_relation_mapping, f, indent=4)


def save_cluster_embedding(entity_embeddings, relation_embeddings):
    with open(f"{output_dir}/cluster_entity_embeddings.pkl", 'wb') as f:
        pickle.dump(entity_embeddings, f)
    with open(f"{output_dir}/cluster_relation_embeddings.pkl", 'wb') as f:
        pickle.dump(relation_embeddings, f)


def find_optimal_threshold(embeddings, min_threshold=0.05, max_threshold=0.37, num_thresholds=32, sample_size=40000, specified_threshold=None):
    if specified_threshold is None:
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
        best_score = -1.0
        best_threshold = None
        best_labels = None

        # Randomly select a sample of entity names
        sample_entities = np.random.choice(list(embeddings.keys()), size=min(sample_size, len(embeddings)), replace=False)
        sample_embeddings = [embeddings[entity] for entity in sample_entities]

        def evaluate_threshold(threshold):
            print(f"Evaluating threshold: {threshold:.2f}")
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average', metric='cosine')
            labels = clustering.fit_predict(sample_embeddings)
            score = silhouette_score(sample_embeddings, labels)

            # Create mappings for the current threshold
            new_to_original_mapping, _ = create_mappings(labels, sample_entities, embeddings)

            # Save the new_to_original mapping for the current threshold
            output_dir = "./cache/2_kg_construct/cluster/"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/new_to_original_mapping_threshold_{threshold:.2f}.json", 'w') as f:
                json.dump(new_to_original_mapping, f, indent=4)

            return threshold, score, labels

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(evaluate_threshold, threshold) for threshold in thresholds]
            for future in as_completed(futures):
                threshold, score, labels = future.result()
                print(f"Threshold: {threshold:.2f}, Silhouette Score: {score:.3f}")
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_labels = labels
        print(f"Best threshold: {best_threshold:.2f}, Best Silhouette Score: {best_score:.3f}.")
    else:
        best_threshold = specified_threshold

    print(f"Now evaluating the best threshold on the full dataset with threshold: {best_threshold:.2f}")
    # Assign labels to the full dataset using the best threshold
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=best_threshold, linkage='average', metric='cosine')
    best_labels = clustering.fit_predict(list(embeddings.values()))

    return best_threshold, best_labels, clustering


def find_cluster_representative(cluster_items, embeddings_dict):
    cluster_embeddings = [embeddings_dict[item] for item in cluster_items]
    cluster_center = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
    representative_index = np.argmin(distances)
    return cluster_items[representative_index]


def create_mappings(labels, items, embeddings_dict):
    new_to_original_mapping = {}
    original_to_new_mapping = {}

    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_items = [items[idx] for idx in cluster_indices]
        representative_item = find_cluster_representative(cluster_items, embeddings_dict)
        new_to_original_mapping[representative_item] = cluster_items
        for item in cluster_items:
            original_to_new_mapping[item] = representative_item

    return new_to_original_mapping, original_to_new_mapping


def compute_cluster_embeddings(mapping, embeddings):
    cluster_embeddings = {}
    for new_item, original_items in mapping.items():
        cluster_embeddings[new_item] = np.mean([embeddings[item] for item in original_items], axis=0).tolist()
    return cluster_embeddings


if __name__ == '__main__':
    kg_path = "./graph/kg_raw.txt"
    clustered_kg_path = "./graph/kg_cluster.txt"

    entity_embeddings, relation_embeddings = load_embedding()

    # entity cluster
    print("Finding optimal threshold for entity clustering...")
    entity_threshold, entity_labels, entity_clustering_res = find_optimal_threshold(entity_embeddings, specified_threshold=ENT_THRESHOLD)
    print(f"Optimal threshold for entity clustering: {entity_threshold:.2f}")

    # entity cluster<->origin mapping
    print(f"Creating mappings between new and original entities...")
    entities_ = list(entity_embeddings.keys())
    new_to_original_entity_mapping, original_to_new_entity_mapping = create_mappings(entity_labels, entities_, entity_embeddings)

    # relation cluster
    print("Finding optimal threshold for relation clustering...")
    relation_threshold, relation_labels, relation_clustering_res = find_optimal_threshold(relation_embeddings, specified_threshold=REL_THRESHOLD)
    print(f"Optimal threshold for relation clustering: {relation_threshold:.2f}")

    # relation cluster<->origin mapping
    print(f"Creating mappings between new and original relations...")
    relations_ = list(relation_embeddings.keys())
    new_to_original_relation_mapping, original_to_new_relation_mapping = create_mappings(relation_labels, relations_, relation_embeddings)

    save_mapping(original_to_new_entity_mapping,
                 new_to_original_entity_mapping,
                 original_to_new_relation_mapping,
                 new_to_original_relation_mapping)

    # cluster embedding
    print("Computing and saving entity cluster embeddings...")
    entity_cluster_embeddings = compute_cluster_embeddings(new_to_original_entity_mapping, entity_embeddings)
    relation_cluster_embeddings = compute_cluster_embeddings(new_to_original_relation_mapping, relation_embeddings)
    save_cluster_embedding(entity_cluster_embeddings, relation_cluster_embeddings)

    # update the knowledge graph
    new_kg_triples = set()
    print("Updating knowledge graph with new entity and relation names...")
    edges = read_graph_from_file("./graph/kg_raw.txt")

    with open(clustered_kg_path, 'w', encoding='utf-8') as f:
        for edge in edges:
            head, relation, tail = edge
            new_head = original_to_new_entity_mapping[head]
            new_tail = original_to_new_entity_mapping[tail]
            new_relation = original_to_new_relation_mapping[relation]

            triple = f"{new_head}\t{new_relation}\t{new_tail}\n"
            if triple not in new_kg_triples:
                new_kg_triples.add(triple)
                f.write(triple)

    print("Clustering completed successfully.")
