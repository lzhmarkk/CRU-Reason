import os
import sys
import math
import json
import random
import multiprocessing
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
from functools import partial
import numpy as np

os.chdir("/ossfs/workspace/EHRLLM")
sys.path.append("/ossfs/workspace/EHRLLM")
from data.dataloader import load_dataset
from utils.utils import set_random_seed
from modules.context import get_concepts
from modules.post_process import get_label_tokenizer

dataset = 'mimic3'
task = 'drugrec'

def handle_patient(item, cond_tokenizer, proc_tokenizer, drug_tokenizer, n_conds):
    id, patient = item
    conditions, procedures, _ = get_concepts(patient, task, with_history=True, flatten=True)

    # 使用tokenizer转换ID
    cond2id = lambda x: cond_tokenizer.convert_tokens_to_indices([x])[0]
    proc2id = lambda x: proc_tokenizer.convert_tokens_to_indices([x])[0]
    drug2id = lambda x: drug_tokenizer.convert_tokens_to_indices([x])[0]

    conditions = [cond2id(x) for x in conditions if x in cond_tokenizer.vocabulary]
    procedures = [proc2id(x) + n_conds for x in procedures if x in proc_tokenizer.vocabulary]
    drugs = [drug2id(x) for x in patient['drugs'] if x in drug_tokenizer.vocabulary]

    conditions = list(set(conditions))
    procedures = list(set(procedures))
    drugs = list(set(drugs))

    features = conditions + procedures
    if len(features) > 28:
        return None, None

    clusters = []
    for cluster_size in [1, 2, 4, 6, 8, 10]:
        if cluster_size > len(features):
            continue
        for cluster in combinations(features, cluster_size):
            cluster_name = "-".join([str(x) for x in sorted(cluster)])
            clusters.append(cluster_name)

    return clusters, drugs

def process_chunk(chunk, cond_tokenizer, proc_tokenizer, drug_tokenizer, n_conds):
    process_id, chunk = chunk
    local_counter = defaultdict(lambda: defaultdict(int))

    print(f"Process {process_id}: Start processing")
    for i, item in enumerate(chunk):
        clusters, drugs = handle_patient(item, cond_tokenizer, proc_tokenizer, drug_tokenizer, n_conds)
        if clusters is None or drugs is None:
            continue

        # 更新计数器
        for cluster in clusters:
            for drug in drugs:
                local_counter[cluster][drug] += 1

        if i % 10 == 9:
            print(f"Process {process_id}: [{i + 1}/{len(chunk)}].")

        # 每处理50个样本清理一次低频项
        clean_every = 25
        low_freq_thres = 3
        if i % clean_every == clean_every - 1 or i == len(chunk) - 1:
            del_count = 0
            for cluster_dict in list(local_counter.values()):
                for drug in list(cluster_dict.keys()):
                    if cluster_dict[drug] <= low_freq_thres:
                        del cluster_dict[drug]
                        del_count += 1
            print(f"Process {process_id}: [{i + 1}/{len(chunk)}]. Cleaned {del_count} low-frequency entries")

    # mean freq
    max_freq = -1
    for cluster, drug_counts in local_counter.items():
        for drug, count in drug_counts.items():
            max_freq = max(max_freq, count)
    print(f"Process {process_id}: Mean frequency: {max_freq}")

    # del entries with freq < mean_freq
    del_count = 0
    for cluster, drug_counts in list(local_counter.items()):
        for drug in list(drug_counts.keys()):
            if drug_counts[drug] < 0.5 * max_freq:
                del drug_counts[drug]
                del_count += 1
    print(f"Process {process_id}: [Final]. Cleaned {del_count} relatively low-frequency entries")

    print(f"Process {process_id}: Finished")
    return {key: dict(value) for key, value in local_counter.items()}

if __name__ == '__main__':
    set_random_seed(42)
    datasets, full_set = load_dataset(dataset, task)
    cond_tokenizer = get_label_tokenizer(full_set, 'conditions')
    proc_tokenizer = get_label_tokenizer(full_set, 'procedures')
    drug_tokenizer = get_label_tokenizer(full_set, 'drugs')

    n_conds = cond_tokenizer.get_vocabulary_size()
    dirs = "./TherapyReason/cache/count/"
    os.makedirs(dirs, exist_ok=True)

    items = [(i, x) for i, x in enumerate(datasets['train'][:5000])]
    random.seed(42)
    random.shuffle(items)

    # 多进程处理
    num_workers = 16
    chunk_size = math.ceil(len(items) / num_workers)
    chunks = [(i // chunk_size, items[i:i+chunk_size]) for i in range(0, len(items), chunk_size)]

    # 创建部分函数固定tokenizer参数
    worker_func = partial(
        process_chunk,
        cond_tokenizer=cond_tokenizer,
        proc_tokenizer=proc_tokenizer,
        drug_tokenizer=drug_tokenizer,
        n_conds=n_conds
    )

    cluster_to_drug = defaultdict(int)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(worker_func, chunks), total=len(chunks), desc="Processing chunks"))

        # 合并结果
        for result in results:
            for cluster, drug_counts in tqdm(result.items()):
                for drug, count in drug_counts.items():
                    cluster_to_drug[f"{cluster}->{drug}"] += count

    # 排序并保存结果
    cluster_to_drug = dict(cluster_to_drug)
    print(f"Total clusters: {len(cluster_to_drug)}")
    cluster_to_drug_sort = sorted(
        cluster_to_drug.items(),
        key=lambda x: x[1],
        reverse=True
    )
    cluster_to_drug_sel = cluster_to_drug_sort[:1000]
    with open("x.json", 'w') as fp:
        json.dump(cluster_to_drug_sel, fp, indent=2)
