import os
import argparse
import numpy as np
from collections import defaultdict

from data.dataloader import load_dataloader
from utils.load_model import get_model
from utils.trainer import MyTrainer as Trainer

args = argparse.ArgumentParser()
args.add_argument("--task", type=str, choices=["mortality", "readmission", "drugrec", "lenofstay", 'diagnosis'], required=True)
args.add_argument("--dataset", type=str, choices=["mimic3", "mimic4"], required=True)
args.add_argument("--method", type=str, required=True)
args.add_argument("--model", type=str, help="Large Language Model")
args.add_argument("--server", type=str, help="API provider for LLM")
args.add_argument("--exp", type=str, help="Experiment name", default="base")
args.add_argument("--embedding_dim", type=int, default=128)
args.add_argument("--batch_size", type=int, default=64)
args.add_argument("--n_workers", type=int, default=8,
                  help="Worker count for LLM-based baseline preprocessing")
config = args.parse_args()

if __name__ == '__main__':
    scores = defaultdict(lambda: defaultdict(list))
    output_path = os.path.join("./saves/", f"{config.dataset}-{config.task}", config.method, config.exp)
    dataset, dataloaders, data_features = load_dataloader(dataset=config.dataset,
                                                          task=config.task,
                                                          batch_size=config.batch_size,
                                                          ratios=None)
    config.data_features = data_features

    for ckpt in filter(lambda _: os.path.isdir(os.path.join(output_path, _)), os.listdir(output_path)):
        ckpt = os.path.join(output_path, str(ckpt), "best.ckpt")
        model_kwargs = {"full_dataset": dataset,
                        "datasets": {"train": dataloaders["train"].dataset, "val": dataloaders["val"].dataset, "test": dataloaders["test"].dataset}}
        evaluator_kwargs = {"checkpoint_path": ckpt, "enable_logging": False}

        if config.task == "mortality" or config.task == "readmission":
            model = get_model(config, mode='binary', feature_keys=["conditions", "procedures", "drugs"], **model_kwargs)
            evaluator = Trainer(model=model, metrics=["pr_auc", "roc_auc", "accuracy", "f1", "jaccard"], **evaluator_kwargs)

        elif config.task == "drugrec":
            model = get_model(config, mode='multilabel', feature_keys=["conditions", "procedures"], **model_kwargs)
            evaluator = Trainer(model=model, metrics=["pr_auc_samples", "roc_auc_samples", "f1_samples", "jaccard_samples"], **evaluator_kwargs)

        elif config.task == "lenofstay":
            model = get_model(config, mode='multiclass', feature_keys=["conditions", "procedures"], **model_kwargs)
            evaluator = Trainer(model=model, metrics=["roc_auc_weighted_ovr", "cohen_kappa", "accuracy", "f1_weighted"], **evaluator_kwargs)

        elif config.task == "diagnosis":
            model = get_model(config, mode='multilabel', feature_keys=["conditions", "procedures", "drugs"], **model_kwargs)
            evaluator = Trainer(model=model, metrics=["pr_auc_samples", "roc_auc_samples", "f1_samples", "jaccard_samples"], **evaluator_kwargs)

        else:
            raise NotImplementedError

        metric_multi = evaluator.evaluate(dataloaders['test'])
        for thres, metric in enumerate(metric_multi):
            for k, v in metric.items():
                scores[thres][k].append(v)

    # output metrics
    for thres in scores:
        score = list(scores[thres].items())
        keys = [e[0][:5] for e in score]
        avg = [round(np.mean(e[1]).item(), 3) for e in score]
        std = [round(np.std(e[1]).item(), 3) for e in score]
        print("\t".join(keys))
        print("\t".join(map(str, avg)))  # avg
        print("\t".join(map(str, std)))  # std
