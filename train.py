import os
import json
import logging
import argparse
import numpy as np
from collections import defaultdict

from data.dataloader import load_dataloader
from utils.utils import set_random_seed
from utils.load_model import get_model
from utils.trainer import MyTrainer as Trainer

args = argparse.ArgumentParser()
args.add_argument("--task", type=str, choices=["mortality", "readmission", "drugrec", "lenofstay", 'diagnosis'], required=True)
args.add_argument("--dataset", type=str, choices=["mimic3", "mimic4"], required=True)
args.add_argument("--method", type=str, required=True)
args.add_argument("--model", type=str, help="Large Language Model")
args.add_argument("--server", type=str, help="API provider for LLM")
args.add_argument("--exp", type=str, help="Experiment name", default="base")
args.add_argument("--train_ratio", type=float, default=1.0)
args.add_argument("--n_runs", type=int, default=10)
args.add_argument("--epochs", type=int, default=50)
args.add_argument("--seed", type=int, default=0)
args.add_argument("--patience", type=int, default=10)
args.add_argument("--embedding_dim", type=int, default=128)
args.add_argument("--batch_size", type=int, default=4)
args.add_argument("--n_workers", type=int, default=8,
                  help="Worker count for LLM-based baseline preprocessing")
args.add_argument("--resume_epoch", type=int, default=-1)
config = args.parse_args()

if __name__ == '__main__':
    logger = logging.getLogger("pyhealth.trainer")
    scores = defaultdict(list)
    set_random_seed(0)
    output_path = os.path.join("./saves/", f"{config.dataset}-{config.task}", config.method, config.exp)
    dataset, dataloaders, data_features = load_dataloader(dataset=config.dataset,
                                                          task=config.task,
                                                          batch_size=config.batch_size,
                                                          ratios=None,
                                                          train_ratio=config.train_ratio,
                                                          n_train_samples=-1, n_val_samples=-1)
    config.data_features = data_features

    for run_id in range(config.n_runs):
        run_id = run_id + config.seed
        set_random_seed(run_id)

        model_kwargs = {"full_dataset": dataset,
                        "datasets": {"train": dataloaders["train"].dataset, "val": dataloaders["val"].dataset, "test": dataloaders["test"].dataset}}
        trainer_kwargs = {"output_path": output_path, "exp_name": f"run{run_id}"}
        train_kwargs = {"train_dataloader": dataloaders['train'],
                        "val_dataloader": dataloaders['val'],
                        "epochs": config.epochs, "patience": config.patience,
                        "resume_epoch": config.resume_epoch,
                        "optimizer_params": {"lr": 1e-5, 'weight_decay': 1e-5}}

        if config.task == "mortality" or config.task == "readmission":
            model = get_model(config, mode='binary', feature_keys=["conditions", "procedures", "drugs"], **model_kwargs)
            trainer = Trainer(model=model, metrics=["pr_auc", "roc_auc", "accuracy", "f1", "jaccard"], **trainer_kwargs)
            trainer.train(monitor="roc_auc",
                          monitor_criterion='max',
                          **train_kwargs)

        elif config.task == "drugrec":
            model = get_model(config, mode='multilabel', feature_keys=["conditions", "procedures"], **model_kwargs)
            trainer = Trainer(model=model, metrics=["pr_auc_samples", "roc_auc_samples", "f1_samples", "jaccard_samples"], **trainer_kwargs)
            trainer.train(monitor="roc_auc_samples",
                          monitor_criterion='max',
                          **train_kwargs)

        elif config.task == "lenofstay":
            model = get_model(config, mode='multiclass', feature_keys=["conditions", "procedures"], **model_kwargs)
            trainer = Trainer(model=model, metrics=["roc_auc_weighted_ovr", "cohen_kappa", "accuracy", "f1_weighted"], **trainer_kwargs)
            trainer.train(monitor="roc_auc_weighted_ovr",
                          monitor_criterion='max',
                          **train_kwargs)

        elif config.task == "diagnosis":
            model = get_model(config, mode='multilabel', feature_keys=["conditions", "procedures", "drugs"], **model_kwargs)
            trainer = Trainer(model=model, metrics=["pr_auc_samples", "roc_auc_samples", "f1_samples", "jaccard_samples"], **trainer_kwargs)
            trainer.train(monitor="roc_auc_samples",
                          monitor_criterion='max',
                          **train_kwargs)

        else:
            raise NotImplementedError

        score = trainer.evaluate(dataloaders['test'])
        for k, v in score[0].items():
            scores[k].append(v)

        logger.removeHandler(logger.handlers[-1])

    # output metrics
    scores = list(scores.items())
    keys = [e[0] for e in scores]
    avg = [round(np.mean(e[1]).item(), 4) for e in scores]
    std = [round(np.std(e[1]).item(), 4) for e in scores]
    logger.info("\t".join(keys))
    logger.info("\t".join(map(str, avg)))  # avg
    logger.info("\t".join(map(str, std)))  # std
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump({"mean": {k: v for k, v in zip(keys, avg)}, "std": {k: v for k, v in zip(keys, std)}}, f, indent=6)
