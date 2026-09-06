import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from sklearn.metrics import f1_score
from utils.llm import load_model
from modules.post_process import POST_PROCESS_FN


def validate_response(prediction, label, threshold=3):
    pos_label, neg_label = label

    pred, golden = [], []
    for drugs, _golden in [(pos_label, 1), (neg_label, 0)]:
        for drug in drugs:
            try:
                score = int(prediction[drug]['score'])
            except:
                score = 0
            pred.append(int(score >= threshold))
            golden.append(_golden)

    return pred, golden


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--val_file", type=str, required=True)
    args.add_argument("--save_dir", type=str, required=True)

    args.add_argument("--task", type=str, choices=['drugrec', 'diagnosis'], required=True)

    args.add_argument("-model", "--model_name_or_path", type=str, help="LLM Checkpoint", required=True)

    config = args.parse_args()
    os.makedirs(config.save_dir, exist_ok=True)

    # `utils.llm.load_model` returns the project's LLM wrapper
    model = load_model({"model": config.model_name_or_path, "server": "local"})

    datasets = load_dataset("json", data_files={"val": config.val_file})
    val_dataset = datasets['val']

    results = defaultdict(list)
    for i, example in enumerate(tqdm(val_dataset, desc="Evaluating")):
        labels = (example['pos_label'], example['neg_label'])

        cache_file = Path(os.path.join(config.save_dir, f"{i}.json"))
        if cache_file.exists():
            prediction = json.loads(cache_file.read_text())['prediction']

        else:
            prompt = example['input']

            try:
                _, (response, prediction), _, usage = model.understand(
                    prompt,
                    post_process=POST_PROCESS_FN[config.task],
                    do_sample=False,
                    max_new_tokens=8192,
                )
            except:  # In early training stages, checkpoints sometimes fail to output valid outputs
                print(cache_file)
                continue

            cache = {"response": response, "prediction": prediction, "prompt": prompt}
            cache_file.write_text(json.dumps(cache, indent=4))

        for threshold in range(1, 5):
            pred, label = validate_response(prediction, labels, threshold=threshold)
            results[threshold].append(f1_score(label, pred))

    print(config.model_name_or_path)
    for threshold, scores in results.items():
        print(threshold, np.mean(scores))
