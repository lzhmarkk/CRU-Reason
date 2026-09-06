import numpy as np
from sklearn.metrics import (average_precision_score,
                             roc_auc_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             jaccard_score,
                             cohen_kappa_score)


def mortality_metrics(y_true_all, y_prob_all, threshold=0.5, verbose=False):
    # y_true_all: List[int], y_prob_all: List[float]
    y_pred_all = [int(x >= threshold) for x in y_prob_all]

    metrics = {
        "avg_prob": np.mean(y_prob_all),
        "pr_auc": average_precision_score(y_true_all, y_prob_all),
        "roc_auc": roc_auc_score(y_true_all, y_prob_all),
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "f1": f1_score(y_true_all, y_pred_all),
        "jaccard": jaccard_score(y_true_all, y_pred_all)
    }

    metrics = {k: round(v, 3) for k, v in metrics.items()}
    if verbose:
        print("avg prob\t", ["avg_prob"])
        print("auprc\tauroc\taccu\tf1   \tjaccard")
        print(f"{metrics['pr_auc']}\t{metrics['roc_auc']}\t{metrics['accuracy']}\t{metrics['f1']}\t{metrics['jaccard']}")

    return metrics


def lenofstay_metrics(y_true_all, y_prob_all, verbose=False):
    # y_true_all: List[int], y_prob_all: List[float]
    y_pred_all = [np.argmax(x) for x in y_prob_all]

    metrics = {
        "avg_prob": np.mean(np.max(y_prob_all, axis=1)),
        "roc_auc_weighted_ovr": roc_auc_score(y_true_all, y_prob_all, average="weighted", multi_class="ovr"),
        "cohen_kappa": cohen_kappa_score(y_true_all, y_pred_all),
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "f1_weighted": f1_score(y_true_all, y_pred_all, average="weighted")
    }

    metrics = {k: round(v, 3) for k, v in metrics.items()}
    if verbose:
        print("avg prob\t", metrics["avg_prob"])
        print("auroc\tkappa\taccu\tf1")
        print(f"{metrics['roc_auc_weighted_ovr']}\t{metrics['cohen_kappa']}\t{metrics['accuracy']}\t{metrics['f1_weighted']}")

    return metrics


def drugrec_metrics(y_true_all, y_prob_all, threshold=0.5, verbose=False):
    # y_true_all: List[int], y_prob_all: List[float]
    N, D = y_prob_all.shape
    y_pred_all = (y_prob_all > threshold).astype(int)

    # LLM-based methods output discrete probabilities
    # So for fairness we round all probabilities to 1 decimal
    y_prob_all_bin = y_prob_all.round(1)

    hamming_loss = np.mean(np.not_equal(y_true_all, y_pred_all))

    eps = 1e-10
    sample_rankings = []
    for i in range(N):
        sorted_indices = sorted(range(D), key=lambda j: -y_prob_all[i][j] - eps * y_true_all[i][j])
        rankings = np.empty((D,))
        rankings[sorted_indices] = range(D)
        true_rankings = rankings[y_true_all[i] == 1]
        norm_true_rankings = np.mean(1 + true_rankings) / np.mean(1 + np.arange(y_true_all[i].sum()))
        sample_rankings.append(norm_true_rankings)
    avg_rank = np.mean(sample_rankings)

    metrics = {
        "avg_prob": np.mean(y_prob_all).round(3),
        "pr_auc_samples": average_precision_score(y_true_all, y_prob_all, average="samples"),
        "pr_auc_samples_bin": average_precision_score(y_true_all, y_prob_all_bin, average="samples"),
        "roc_auc_samples": roc_auc_score(y_true_all, y_prob_all, average="samples"),
        "roc_auc_samples_bin": roc_auc_score(y_true_all, y_prob_all_bin, average="samples"),
        "accuracy": accuracy_score(y_true_all.flatten(), y_pred_all.flatten()),
        "precision_samples": precision_score(y_true_all, y_pred_all, average="samples"),
        "recall_samples": recall_score(y_true_all, y_pred_all, average="samples"),
        "f1_samples": f1_score(y_true_all, y_pred_all, average="samples"),
        "jaccard_samples": jaccard_score(y_true_all, y_pred_all, average="samples"),
        "hamming_loss": hamming_loss,
        "rank": avg_rank
    }

    metrics = {k: round(v, 3) for k, v in metrics.items()}
    if verbose:
        print("avg prob\t", metrics["avg_prob"])
        print("auprc \tauroc\tf1\tjaccard")
        print(f"{metrics['accuracy']}\t{metrics['precision_samples']}\t"
              f"{metrics['recall_samples']}\t{metrics['f1_samples']}\t{metrics['jaccard_samples']}"
              f"{metrics['hamming_loss']}\t{metrics['rank']}")

    return metrics


def get_metrics_fn(mode: str):
    if mode == "binary" or mode == 'mortality' or mode == 'readmission':
        return mortality_metrics
    elif mode == "multiclass" or mode == 'lenofstay':
        return lenofstay_metrics
    elif mode == "multilabel" or mode == 'drugrec' or mode == 'diagnosis':
        return drugrec_metrics
    else:
        raise ValueError(f"Mode/Task {mode} is not supported")
