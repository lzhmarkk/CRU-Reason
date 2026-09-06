import os
import json
import torch
from tqdm import tqdm
from graph.embedding.embedding import generate_embeddings


def patient_base_context(sample, task):
    patient_id = sample['patient_id']
    visit_id = sample['visit_id']

    output = f"Patient ID: {patient_id}\n"

    feature_keys = ['conditions', 'procedures'] if task in ['drugrec'] \
        else ['conditions', 'procedures', 'drugs']
    assert all(k in sample for k in feature_keys)

    n_visits = len(sample['conditions'])
    feature_set = {k: set() for k in feature_keys}

    for visit_id in range(n_visits):
        output += f"Visit {visit_id + 1}:\n"

        feature_count = {k: 1 for k in feature_keys}
        for feature_key in feature_keys:
            output += f"{feature_key.capitalize()}:\n"
            for concept in sample[feature_key][visit_id]:
                output += f"{feature_count[feature_key]}. {concept}"
                if concept in feature_set[feature_key]:
                    output += f" (continued from previous visit)\n"
                elif visit_id != 0:
                    output += f" (new)\n"
                else:
                    output += f"\n"

                feature_set[feature_key].add(concept)
                feature_count[feature_key] += 1

        output += "\n"

    return output.strip()


def get_patient_embedding(sample_dataset, output_dir, task):
    def truncate_text(text, max_length):
        if len(text) > max_length:
            truncated = text[:max_length]
            return truncated, True
        return text, False

    output_dir = os.path.join(output_dir, "patient_embedding")
    os.makedirs(output_dir, exist_ok=True)

    # patient context & embedding
    contexts, embeddings = {}, {}
    id2idx, idx2id = {}, {}

    for idx, sample in tqdm(enumerate(sample_dataset), total=len(sample_dataset)):
        sample_id = f"{sample['patient_id']}_{sample['visit_id']}"
        output_file = os.path.join(output_dir, f"{sample_id}.json")

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text, embedding = data['text'], data['embedding']

        else:
            text = patient_base_context(sample, task)
            truncated_text, _ = truncate_text(text, max_length=30000)

            embedding = generate_embeddings([truncated_text])
            embedding = list(embedding.values())[0]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"text": text, "embedding": embedding}, f)

        contexts[sample_id] = text
        embeddings[sample_id] = torch.tensor(embedding)
        id2idx[sample_id] = idx
        idx2id[idx] = sample_id

    return contexts, embeddings, id2idx, idx2id
