import os
import json
import faiss
import numpy as np
from tqdm import tqdm

MAX_CONTEXT_LENGTH = 20000


def get_faiss_similarity(sample_dataset, contexts, embeddings, id2idx, idx2id, task):
    MAX_K = 2
    label_key = {"mortality": "label", 'readmission': "label", 'lenofstay': 'label',
                 'drugrec': 'drugs', 'diagnosis': "label"}[task]

    def is_context_valid(context):
        return len(context) <= MAX_CONTEXT_LENGTH

    # faiss index
    sample_ids = [idx2id[idx] for idx in range(len(sample_dataset))]

    print("Building Faiss index...")
    embedding_matrix = np.array([embeddings[sid] for sid in sample_ids]).astype('float32')
    faiss.normalize_L2(embedding_matrix)
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])  # Inner product for normalized vectors is cosine similarity
    index.add(embedding_matrix)

    print("Searching for top-K similar samples...")
    target_patient_ids = sample_ids

    similar_patients = {}
    for sample_id in tqdm(target_patient_ids):
        assert sample_id in id2idx, f"Sample ID {sample_id} not found in embeddings."

        if len(contexts[sample_id]) > 30000:
            continue

        idx = id2idx[sample_id]
        label = sample_dataset[idx][label_key]
        current_embedding = embedding_matrix[idx].reshape(1, -1)

        # Search for the nearest neighbors
        D, I = index.search(current_embedding, 100)  # Search for more neighbors to filter later

        neighbor_ids = []
        for neighbor_idx in I[0]:
            neighbor_id = idx2id[neighbor_idx]
            if neighbor_id == sample_id or neighbor_id.split("_")[0] == sample_id.split("_")[0]:
                continue  # Skip the same patient or other instances of the same patient
            neighbor_ids.append(neighbor_id)
            if len(neighbor_ids) >= 50:  # Collect enough neighbors to find top-K after filtering
                break

        # Compute similarity scores for the neighbors
        similarity_scores = {pid: np.dot(current_embedding[0], embedding_matrix[id2idx[pid]]) for pid in neighbor_ids}

        # Separate positive and negative patients
        if task in ['mortality', 'readmission', 'lenofstay']:
            positive_ids = [pid for pid in similarity_scores if sample_dataset[id2idx[pid]][label_key] == label]
            negative_ids = [pid for pid in similarity_scores if sample_dataset[id2idx[pid]][label_key] != label]
        else:
            get_overlap = lambda a, b: len(set(a).intersection(set(b))) / min(len(a), len(b))
            positive_ids = [pid for pid in similarity_scores if get_overlap(sample_dataset[id2idx[pid]][label_key], label) >= 0.5]
            negative_ids = []

        # Sort and select top-K positive and negative patients
        sorted_positive_ids = sorted(positive_ids, key=lambda pid: similarity_scores[pid], reverse=True)
        # Get up to MAX_K positive patients
        sorted_positive_ids = [pid for pid in sorted_positive_ids if is_context_valid(contexts[pid])][:MAX_K]

        sorted_negative_ids = sorted(negative_ids, key=lambda pid: similarity_scores[pid], reverse=True)
        # Get up to MAX_K negative patients
        sorted_negative_ids = [pid for pid in sorted_negative_ids if is_context_valid(contexts[pid])][:MAX_K]

        similar_patients[sample_id] = {"positive": sorted_positive_ids, "negative": sorted_negative_ids}

    return similar_patients


def get_faiss_similarity_cached(sample_dataset, contexts, embeddings, id2idx, idx2id, task, output_dir):
    output_file = os.path.join(output_dir, "patient_similarity.json")
    if os.path.exists(output_file):
        with (open(output_file, 'r', encoding='utf-8') as f):
            similar_patients = json.load(f)

    else:
        similar_patients = get_faiss_similarity(sample_dataset, contexts, embeddings, id2idx, idx2id, task)

        # Save updated communities to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(similar_patients, f)

    return similar_patients
