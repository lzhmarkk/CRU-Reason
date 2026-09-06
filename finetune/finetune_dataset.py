import os
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from data.dataloader import load_dataloader
from finetune.CoT.CoT import CoTFineTune
from finetune.KARE.KARE import KAREFineTune

args = argparse.ArgumentParser()
args.add_argument("--task", type=str, choices=["mortality", "readmission", "drugrec", "lenofstay", 'diagnosis'])
args.add_argument("--dataset", type=str, choices=["mimic3", "mimic4"])

args.add_argument("--method", type=str, help="Algorithm")
args.add_argument("--model", type=str, help="Large Language Model")
args.add_argument("--genre", type=str, help="Genre of LLM")
args.add_argument("--server", type=str, help="API provider for LLM")
args.add_argument("--max_attempts", type=int, default=5)

args.add_argument("--batch_size", type=int, default=512)
args.add_argument("--n_workers", type=int, default=8)
args.add_argument("--n_train_samples", type=int, default=8000, help="Number of training samples, -1 for all")
args.add_argument("--n_val_samples", type=int, default=500, help="Number of training samples, -1 for all")
config = args.parse_args()


def get_model(config, mode, feature_keys=None, full_dataset=None, datasets=None, **kwargs):
    if feature_keys is None: feature_keys = ["conditions", "procedures", "drugs"]

    kwargs.update({"dataset": full_dataset,
                   "feature_keys": feature_keys,
                   "label_key": "drugs" if config.task == 'drugrec' else "label",
                   "mode": mode,
                   "config": config.__dict__})
    kwargs['train_set'] = datasets['train']
    kwargs['val_set'] = datasets['val']
    kwargs['test_set'] = datasets['test']

    if config.method == 'CoT':
        model = CoTFineTune(**kwargs)
    elif config.method == 'KARE':
        model = KAREFineTune(**kwargs)
    else:
        raise ValueError

    return model


if __name__ == '__main__':
    scores = defaultdict(lambda: defaultdict(list))
    output_path = os.path.join("./saves/", f"{config.dataset}-{config.task}", config.method)
    dataset, dataloaders, data_features = load_dataloader(dataset=config.dataset,
                                                          task=config.task,
                                                          batch_size=config.batch_size,
                                                          n_train_samples=config.n_train_samples,
                                                          n_val_samples=config.n_val_samples)
    config.data_features = data_features

    model_kwargs = {"full_dataset": dataset,
                    "datasets": {"train": dataloaders["train"].dataset, "val": dataloaders["val"].dataset, "test": dataloaders["test"].dataset}}
    if config.task == "mortality" or config.task == "readmission":
        model = get_model(config, mode='binary', feature_keys=["conditions", "procedures", "drugs"], **model_kwargs)
    elif config.task == "drugrec":
        model = get_model(config, mode='multilabel', feature_keys=["conditions", "procedures"], **model_kwargs)
    elif config.task == "lenofstay":
        model = get_model(config, mode='multiclass', feature_keys=["conditions", "procedures"], **model_kwargs)
    elif config.task == "diagnosis":
        model = get_model(config, mode='multilabel', feature_keys=["conditions", "procedures", "drugs"], **model_kwargs)
    else:
        raise NotImplementedError

    train_samples, val_samples = [], []
    model.eval()

    for data in tqdm(dataloaders['train'], desc="Generate Training Set"):
        train_samples.extend(model.generate_dataset(split='train', **data))
    print(f"Total {len(train_samples)} train samples")
    train_file = Path(os.path.join(model.cache_dirs['train'], 'corpus.jsonl'))
    train_file.write_text("\n".join([json.dumps(sample) for sample in train_samples]))

    for data in tqdm(dataloaders['val'], desc="Generate Validation Set"):
        val_samples.extend(model.generate_dataset(split='val', **data))
    print(f"Total {len(val_samples)} val samples")
    val_file = Path(os.path.join(model.cache_dirs['val'], 'corpus.jsonl'))
    val_file.write_text("\n".join([json.dumps(sample) for sample in val_samples]))
