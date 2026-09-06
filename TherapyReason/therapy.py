import os
import json
import torch
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from modules.context import get_concepts
from modules.kg import graph
from modules.candidate import load_drugrec_candidate_model, get_candidate_drugs, code2drug, load_diagnosis_candidate_model, get_candidate_diagnosis
from modules.motif import MOTIF_RULE0
from TherapyReason.utils.merge import get_connected_subgraphs
from TherapyReason.utils.significance import sort_by_significance


def get_dl_candidates(thres, cache_dir, dataset, task, full_set, test_set, model_name):
    assert task in ['drugrec', 'diagnosis']
    label_key = 'drugs' if task == 'drugrec' else 'label'
    load_model_fn = load_drugrec_candidate_model if task == 'drugrec' else load_diagnosis_candidate_model

    dl_candidate_file = Path(os.path.join(cache_dir, f"dl_thres{int(thres * 10)}.jsonl"))
    if dl_candidate_file.exists():
        sample_dl_candidates = [json.loads(l) for l in dl_candidate_file.read_text().splitlines()]
    else:
        sample_dl_candidates = []
        dl_model = load_model_fn(dataset, task, full_set, model_name)
        for sample in tqdm(test_set):
            conditions, procedures, drugs = get_concepts(sample, task, with_history=True, flatten=False)
            kwargs = {'conditions': [conditions], 'procedures': [procedures], 'drugs': [drugs]}
            kwargs[label_key] = [sample[label_key]]
            with torch.no_grad():
                output = dl_model(**kwargs)

            solution_score = output['y_prob'][0].cpu().numpy()  # (n_drugs_total,)
            thresholded_indices = np.where(solution_score > thres)[0]
            candidates = dl_model.label_tokenizer.convert_indices_to_tokens(thresholded_indices)
            if model_name in ['MICRON', 'GAMENet', 'MoleRec', 'SafeDrug']:
                candidates = [code2drug[code] for code in candidates]  # ATC codes -> drug names
            sample_dl_candidates.append(candidates)

        dl_candidate_file.write_text("\n".join(json.dumps(x) for x in sample_dl_candidates))

    print(np.mean([len(x) for x in sample_dl_candidates]))
    return sample_dl_candidates


def rank_dl_candidates(cache_dir, dataset, task, full_set, test_set, model_name):
    assert task in ['drugrec', 'diagnosis']
    label_key = 'drugs' if task == 'drugrec' else 'label'
    load_model_fn = load_drugrec_candidate_model if task == 'drugrec' else load_diagnosis_candidate_model

    dl_ranked_file = Path(os.path.join(cache_dir, f"dl_rank.jsonl"))
    if dl_ranked_file.exists():
        sample_dl_ranks = [json.loads(l) for l in dl_ranked_file.read_text().splitlines()]
    else:
        sample_dl_ranks = []
        dl_model = load_model_fn(dataset, task, full_set, model_name)
        for sample in tqdm(test_set):
            conditions, procedures, drugs = get_concepts(sample, task, with_history=True, flatten=False)
            kwargs = {'conditions': [conditions], 'procedures': [procedures], 'drugs': [drugs]}
            kwargs[label_key] = [sample[label_key]]
            with torch.no_grad():
                output = dl_model(**kwargs)

            solution_score = output['y_prob'][0].cpu().numpy()
            sorted_indices = np.argsort(-solution_score)
            all_candidates = dl_model.label_tokenizer.convert_indices_to_tokens(sorted_indices)
            if model_name in ['MICRON', 'GAMENet', 'MoleRec', 'SafeDrug']:
                all_candidates = [code2drug[code] for code in all_candidates]  # ATC codes -> drug names
            sample_dl_ranks.append(all_candidates)

            dl_ranked_file.write_text("\n".join(json.dumps(x) for x in sample_dl_ranks))

    print(np.mean([len(x) for x in sample_dl_ranks]))
    return sample_dl_ranks


def get_kg_candidates(cache_dir, task, test_set, drug_tokenizer, label_tokenizer):
    assert task in ['drugrec', 'diagnosis']

    kg_candidate_file = Path(os.path.join(cache_dir, "kg_1hop.jsonl"))
    if kg_candidate_file.exists():
        sample_kg_candidates = [json.loads(l) for l in kg_candidate_file.read_text().splitlines()]
    else:
        sample_kg_candidates = []
        for sample in tqdm(test_set):
            conditions, procedures, drugs = get_concepts(sample, task, with_history=True, flatten=True)
            candidate_fn = get_candidate_drugs if task == 'drugrec' else get_candidate_diagnosis
            kg_candidates = candidate_fn(fn='kg', model=None, label_tokenizer=label_tokenizer, topk=1000,
                                                concepts=(conditions, procedures, drugs))
            if task == 'diagnosis':
                kg_candidates = kg_candidates + [cond for cond in conditions if cond not in kg_candidates]

            sample_kg_candidates.append(kg_candidates)
        kg_candidate_file.write_text("\n".join(json.dumps(x) for x in sample_kg_candidates))
    sample_kg_candidates = [[x for x in kg_candidates if x in label_tokenizer.vocabulary] for kg_candidates in sample_kg_candidates]
    print(np.mean([len(x) for x in sample_kg_candidates]))
    return sample_kg_candidates


def get_solutions(sample, targets, task, significance, label_key, n_hop, relevant_kwargs=None):
    assert task in ['drugrec', 'diagnosis'], "Other task not implemented"

    cond_visits, proc_visits, drug_visits = get_concepts(sample, task=task, with_history=True, flatten=False)
    solutions_visits = []
    solution_candidates = set()
    n_visits = len(cond_visits)
    for vid in range(n_visits):
        cond, proc, drug = cond_visits[vid], proc_visits[vid], drug_visits[vid]
        if task == 'drugrec':
            assert len(drug) == 0

        queries = sort_by_significance(cond, significance)
        included_nodes = [graph.entity2id[n] for n in cond + proc + drug + targets if n in graph.entities]
        sharable_nodes = proc + targets if task == 'drugrec' else proc + drug
        relevant_kwargs["included_nodes"] = included_nodes
        solutions = get_connected_subgraphs(queries=[graph.entity2id[q] for q in queries if q in graph.entities],
                                            included_nodes=included_nodes,
                                            sharable_nodes=sharable_nodes,
                                            max_path_length=n_hop,
                                            relevant_kwargs=relevant_kwargs,
                                            merge_method=None)
        solutions_visits.append(solutions)

        for solution in solutions:
            conditions = solution['conditions']
            procedures = solution['procedures']
            drugs = solution['drugs']

            solution['conditions'] = [x for x in solution['conditions'] if x in cond]
            solution['procedures'] = [x for x in solution['procedures'] if x in proc]
            solution['drugs'] = [x for x in solution['drugs'] if x in drug]

            if task == 'drugrec':
                solution['candidates'] = [x for x in drugs if x in targets]
            else:
                solution['candidates'] = [x for x in conditions if x in targets]

        # identify unvisited candidates
        for solution in solutions:
            solution_candidates.update(solution['candidates'])

    unvisited_candidates = set(targets) - solution_candidates

    return {
        "solutions": solutions_visits,
        label_key: sample[label_key],
        "isolates": list(unvisited_candidates)
    }


def get_solutions_single(args):
    i, sample, targets, solution_file, task, significance, label_key, n_hop, relevant_kwargs = args
    output = get_solutions(sample, targets, task, significance, label_key, n_hop=n_hop, relevant_kwargs=relevant_kwargs)
    return output


def get_therapy(solution_file, task, significance, label_key, test_set,
                sample_dl_candidates, sample_kg_candidates,
                relevant_kwargs=None):
    if solution_file.exists():
        sample_solutions = [json.loads(l) for l in solution_file.read_text().splitlines()]
        return sample_solutions

    else:
        if task == 'drugrec':
            n_hop = 3
        elif task == 'diagnosis':
            n_hop = 2
        else:
            raise ValueError("Other task not implemented")

        if relevant_kwargs is None:
            relevant_kwargs = {"motifs": MOTIF_RULE0,
                               "use_semantic": True,
                               "use_embedding": False,
                               "use_edge_rule": True}

        items = []
        for i in range(len(test_set)):
            items.append((i, test_set[i], sample_kg_candidates[i],
                          solution_file, task, significance, label_key, n_hop, relevant_kwargs))

        # main therapy construction function
        with multiprocessing.Pool(processes=8) as pool:
            sample_solutions = list(tqdm(pool.imap(get_solutions_single, items), total=len(items)))

        # post construction
        for i in range(len(test_set)):
            solution_obj = sample_solutions[i]
            dl_candidates = sample_dl_candidates[i]
            kg_candidates = sample_kg_candidates[i]

            for visit in solution_obj['solutions']:
                for solution in visit:
                    dl_nodes = [x for x in solution['candidates'] if x in dl_candidates]
                    not_dl_nodes = [x for x in solution['candidates'] if x not in dl_candidates and x in kg_candidates]
                    assert all(x in dl_candidates for x in dl_nodes)
                    assert all(x in kg_candidates for x in not_dl_nodes)

                    solution['dl_candidates'] = dl_nodes
                    solution['kg_candidates'] = not_dl_nodes

        solution_file.write_text("\n".join(json.dumps(x) for x in sample_solutions))

        return sample_solutions
