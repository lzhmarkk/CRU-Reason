"""Training-set disease ranking used by CRU construction (Appendix B.1)."""
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from data.dataloader import load_dataset


def disease_scores(samples, task):
    """Higher score means lower target entropy; count each pair once per sample."""
    if task not in ("drugrec", "diagnosis"):
        raise ValueError("Disease entropy ranking supports drugrec and diagnosis")
    label_key = "drugs" if task == "drugrec" else "label"
    counts = defaultdict(Counter)
    for sample in samples:
        diseases = set(itertools.chain.from_iterable(sample["conditions"]))
        targets = set(sample[label_key])
        for disease in diseases:
            counts[disease].update(targets)
    scores = {}
    for disease, distribution in counts.items():
        total = sum(distribution.values())
        if total:
            scores[disease] = math.fsum(
                (count / total) * math.log(count / total)
                for _, count in sorted(distribution.items())
            )
    return scores


def get_significance(dataset, task):
    path = Path(f"TherapyReason/cache/{dataset}-{task}/simple/significance.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    datasets, _ = load_dataset(dataset, task)
    scores = disease_scores(datasets["train"], task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores), encoding="utf-8")
    return scores


def sort_by_significance(concepts, significance):
    return sorted(concepts, key=lambda disease: (-significance.get(disease, -math.inf), disease))
