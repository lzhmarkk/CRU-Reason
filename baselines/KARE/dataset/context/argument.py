import os
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing
from collections import defaultdict

from graph.embedding.embedding import generate_embeddings

MORTALITY_THEME_TERMS = [
    "terminal condition", "critical diagnosis", "end-stage", "life-threatening",
    "critical interventions", "life-saving measures", "resuscitation", "emergency procedure",
    "palliative medication", "end-of-life drugs", "life support drugs", "emergency meds"
]

READMISSION_THEME_TERMS = [
    "chronic ailment", "postoperative complication", "recurrent", "readmission-prone",
    "follow-up procedure", "secondary intervention", "post-treatment", "treatment review",
    "maintenance medication", "postoperative drugs", "treatment continuation", "follow-up meds"
]

LOS_THEME_TERMS = [
    "prolonged ICU stay", "extended hospitalization", "critical monitoring", "intensive care duration",
    "ventilator dependence", "mechanical ventilation", "hemodynamic instability", "organ support",
    "sedation protocol", "analgesia management", "ICU-acquired condition", "recovery trajectory",
    "step-down readiness", "discharge planning", "resource utilization", "bed occupancy",
    "severity scoring", "clinical deterioration", "rehabilitation potential", "post-ICU recovery"
]

DRUGREC_THEME_TERMS = [
    "therapeutic efficacy", "adverse reaction", "drug interaction", "contraindication",
    "dosage adjustment", "personalized therapy", "pharmacogenomics", "treatment response",
    "polypharmacy risk", "drug substitution", "formulary alternative", "cost-effective option",
    "allergy compatibility", "side effect profile", "patient tolerance", "clinical guideline",
    "off-label use", "drug sensitivity", "therapeutic alternative", "medication adherence"
]

DIAGNOSIS_THEME_TERMS = [
    "early detection", "predictive markers", "clinical indicators", "risk stratification",
    "symptom correlation", "disease progression", "pre-diagnostic signs", "prodromal features",
    "biomarker analysis", "genetic predisposition", "imaging findings", "lab abnormalities",
    "differential diagnosis", "diagnostic uncertainty", "atypical presentation", "comorbid influence",
    "algorithmic prediction", "machine learning features", "predictive modeling", "decision thresholds",
    "false positive risk", "diagnostic delay", "screening yield", "early intervention window"
]


def theme_embedding_cache(output_dir):
    output_path = os.path.join(output_dir, "theme_embedding.pkl")
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            embedding = pickle.load(f)

    else:
        embedding = {"mortality": generate_embeddings(MORTALITY_THEME_TERMS, dimension=None),
                     "readmission": generate_embeddings(READMISSION_THEME_TERMS, dimension=None),
                     "lenofstay": generate_embeddings(LOS_THEME_TERMS, dimension=None),
                     "drugrec": generate_embeddings(DRUGREC_THEME_TERMS, dimension=None),
                     "diagnosis": generate_embeddings(DIAGNOSIS_THEME_TERMS, dimension=None)}

        with open(output_path, "wb") as f:
            pickle.dump(embedding, f)

    return embedding


def theme_relevance(summary_embedding, task, output_dir, lambda_3=0.1):
    theme_embeddings = theme_embedding_cache(output_dir)[task]

    max_similarity = max(np.dot(summary_embedding, theme_embedding) for theme_embedding in theme_embeddings.values())
    return 1 + lambda_3 * max_similarity


def dynamic_graph_retrieval_augmentation(patient_id, patient_nodes, summary_to_nodes, summary_embeddings, patient_base_context_embedding, task,
                                         output_dir,
                                         patient_node_visit_idx=None,
                                         N=10, alpha=0.8, beta=0.5, lambda_1=0.1, lambda_2=0.1, lambda_3=0.1):
    candidate_communities = list(summary_to_nodes.keys())
    selected_summaries = []
    node_hit_count = defaultdict(int)
    patient_direct_nodes = set(patient_nodes["direct"])
    patient_indirect_nodes = set(patient_nodes["indirect"])

    for _ in range(N):
        print(patient_id, _)
        if not candidate_communities:
            break

        relevance_scores = {}
        for summary in candidate_communities:
            community_nodes = set(summary_to_nodes[summary])

            h_direct = len(patient_direct_nodes.intersection(community_nodes)) / len(patient_direct_nodes)

            h_indirect = len(patient_indirect_nodes.intersection(community_nodes)) / len(patient_indirect_nodes)

            decay = np.prod([beta ** node_hit_count[node] for node in community_nodes])

            coherence = 1 + lambda_1 * np.dot(summary_embeddings[summary], patient_base_context_embedding)

            if patient_node_visit_idx:
                hit_direct_nodes = patient_direct_nodes.intersection(community_nodes)
                recency = 1 + lambda_2 * np.mean([patient_node_visit_idx.get(node, 0) for node in hit_direct_nodes])
            else:
                recency = 1

            theme_rel = theme_relevance(summary_embeddings[summary], task, output_dir, lambda_3)

            relevance_score = (h_direct + alpha * h_indirect) * decay * coherence * recency * theme_rel
            relevance_scores[summary] = relevance_score

        best_summary = max(relevance_scores, key=relevance_scores.get)
        selected_summaries.append(best_summary)

        for node in summary_to_nodes[best_summary]:
            node_hit_count[node] += 1

        candidate_communities.remove(best_summary)

    return selected_summaries


def get_argument_context_thread(args):
    patient_id, nodes, base_context_embedding, summary_to_nodes, summary_embeddings, task, output_file, output_dir = args

    print("get_argument_context_thread", patient_id)

    selected_summaries = dynamic_graph_retrieval_augmentation(
        patient_id, nodes, summary_to_nodes, summary_embeddings, base_context_embedding, task, output_dir)

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(selected_summaries, f)

    return patient_id, selected_summaries


def get_argument_context(patient_nodes, patient_context_embs, summary_to_nodes, summary_embeddings, task, output_dir):
    ## Community Relevance Score Computation
    print("Computing relevance scores and augmenting context ...")
    output_dir = os.path.join(output_dir, "argument_context")
    os.makedirs(output_dir, exist_ok=True)

    selected_summaries_all = {}
    items = []
    for patient_id, nodes in patient_nodes.items():
        output_file = os.path.join(output_dir, f"{patient_id}.json")
        if os.path.exists(output_file):
            with open(output_file, "r", encoding='utf-8') as f:
                selected_summaries = json.load(f)
            selected_summaries_all[patient_id] = selected_summaries
        else:
            items.append((patient_id, nodes, patient_context_embs[patient_id], summary_to_nodes, summary_embeddings, task, output_file, output_dir))

    if sys.platform == 'linux':
        num_cpus = min(int(multiprocessing.cpu_count() * 0.8), 64)
        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.map(get_argument_context_thread, items)
    else:
        results = [get_argument_context_thread(item) for item in tqdm(items)]

    for patient_id, selected_summaries in results:
        selected_summaries_all[patient_id] = selected_summaries

    return selected_summaries_all
