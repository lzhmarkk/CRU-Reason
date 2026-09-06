from data.dataloader import load_dataset
from baselines.EMERGE.EMERGE import EMERGE
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from tqdm import tqdm

sample_dataset = load_dataset("mimic4", "readmission")

model = EMERGE(dataset=sample_dataset,
               feature_keys=['conditions', 'procedures', 'drugs'],
               label_key='label',
               mode='binary',
               embedding_dim=128,
               config={
                   "dataset": "mimic4",
                   "task": "mortality",
                   "exp": "base",
                   "model": "deepseek-chat"
               })


def thread_fn(sample):
    patient_id = sample['patient_id'] + "_" + sample['visit_id']

    # summary
    cache_path = os.path.join(model.cache_path, f"{patient_id}.json")
    if os.path.exists(cache_path):
        pass
    else:
        print(patient_id)

        patient = {}
        for feature_key in model.feature_keys:
            patient[feature_key] = sample[feature_key]

        summary, summary_embedding, note, note_embedding = model.get_embedding(patient)

        with open(os.path.join(model.cache_path, f"{patient_id}.json"), "w", encoding='utf-8') as f:
            json.dump({"summary": summary,
                       "summary_embedding": summary_embedding,
                       "note": note,
                       "note_embedding": note_embedding}, f)

    return


if __name__ == '__main__':
    futures = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for sample in sample_dataset:
            future = executor.submit(thread_fn, sample)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
