from baselines.CoT.context.base_context import get_patient_embedding
from baselines.CoT.context.similarity import get_faiss_similarity_cached


def CoT_context(sample_dataset, task, output_dir):
    contexts, embeddings, id2idx, idx2id = get_patient_embedding(sample_dataset, output_dir, task)

    similar_patients = get_faiss_similarity_cached(sample_dataset, contexts, embeddings, id2idx, idx2id, task, output_dir)

    return contexts, similar_patients
