import json
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from data.dataloader import load_mappings
from modules.kg import graph, get_neighbors
from utils.multiprocess import multiprocess_call

from baselines.MKGRank.umls import UMLSBERT
from baselines.MKGRank.prompts import DESCRIPTION_TEMPLATE
from baselines.KARE.dataset.context.argument import MORTALITY_THEME_TERMS, READMISSION_THEME_TERMS, LOS_THEME_TERMS, DRUGREC_THEME_TERMS, \
    DIAGNOSIS_THEME_TERMS


def generate_theme_embedding(task, encoder: UMLSBERT, cache_file):
    if os.path.exists(cache_file):
        embeddings = pickle.loads(Path(cache_file).read_bytes())
    else:
        theme_terms = {
            "mortality": MORTALITY_THEME_TERMS,
            "readmission": READMISSION_THEME_TERMS,
            "lenofstay": LOS_THEME_TERMS,
            "drugrec": DRUGREC_THEME_TERMS,
            "diagnosis": DIAGNOSIS_THEME_TERMS
        }
        theme_terms = theme_terms[task]
        embeddings = encoder.batch_encode(theme_terms)

        Path(cache_file).write_bytes(pickle.dumps(embeddings))

    return embeddings


def get_similarity(query_vecs, rel_vecs, top_k=20):
    sims = cosine_similarity(query_vecs, rel_vecs)
    sims = sims.max(axis=0)
    top_k_indices = np.argsort(-sims)[:top_k].tolist()
    return sims.tolist(), top_k_indices


def get_triplet_cached(theme_embedding, encoder, cache_file):
    if os.path.exists(cache_file):
        all_triplets_dict = json.loads(Path(cache_file).read_text(encoding='utf-8'))

    else:
        all_triplets_dict = {}
        all_cond, all_proc, all_drug = load_mappings()
        all_concepts = list(all_cond.values()) + list(all_proc.values()) + list(all_drug.values())

        for concept in all_concepts:
            triplets = get_neighbors(concept, graph.entities, direction='forward', apply_rule=True)

            if not triplets:
                continue

            relation_texts = [' '.join([rel[0], rel[1].replace('_', ' '), rel[2]]).strip().lower()
                              for rel in triplets]

            relation_embeddings = encoder.batch_encode(relation_texts)

            # Get top-k indices
            sims, top_k_indices = get_similarity(theme_embedding, relation_embeddings, top_k=20)
            top_triples = [triplets[i] for i in top_k_indices]
            top_sims = [sims[i] for i in top_k_indices]

            top_triples = [triple for score, triple in zip(top_sims, top_triples)]

            all_triplets_dict[concept] = top_triples

        Path(cache_file).write_text(json.dumps(all_triplets_dict, indent=4), encoding='utf-8')

    return all_triplets_dict


def generate_description_cached(llm, cache_file):
    def generate_description_single(key, concept):
        query = DESCRIPTION_TEMPLATE.replace("##PLACEHOLDER##", concept)
        _, desc, _, usage = llm.understand(query)
        description = desc.split("Description:")[-1].replace("*", "").strip()
        return key, {"concept": concept, "description": description, "usage": usage.dict()}

    if os.path.exists(cache_file):
        all_descriptions_dict = json.loads(Path(cache_file).read_text(encoding='utf-8'))
    else:
        all_cond, all_proc, all_drug = load_mappings()
        all_concepts = list(all_cond.values()) + list(all_proc.values()) + list(all_drug.values())

        all_descriptions = multiprocess_call(generate_description_single, all_concepts, n_workers=8)

        all_descriptions_dict = {item["concept"]: {"description": item["description"], "usage": item['usage']} for item in all_descriptions}

        Path(cache_file).write_text(json.dumps(all_descriptions_dict), encoding='utf-8')

    return all_descriptions_dict


def generate_concepts_context(task, encoder, llm, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    all_descriptions_dict = generate_description_cached(llm, cache_file=os.path.join(cache_dir, "concept_description.json"))
    all_descriptions = {c: v['description'] for c,v in all_descriptions_dict.items()}
    usage = {c: v['usage'] for c,v in all_descriptions_dict.items()}

    task_cache_dir = os.path.join(cache_dir, task)
    os.makedirs(task_cache_dir, exist_ok=True)

    theme_embedding = generate_theme_embedding(task, encoder, cache_file=os.path.join(task_cache_dir, "theme_embedding.pkl"))

    all_triplets_dict = get_triplet_cached(theme_embedding, encoder, cache_file=os.path.join(task_cache_dir, "triplets.json"))

    return all_descriptions, all_triplets_dict, usage
