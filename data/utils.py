import numpy as np
from collections import defaultdict


def print_statistics(dataset):
    patient_visits = defaultdict(list)
    for sample in dataset:
        patient_visits[sample['patient_id']].append(sample)

    # per patient
    print(f"#patients: {len(patient_visits)}", end="\t")
    print(f"#sample: {len(dataset)}", end="\t")
    print(f"#sample/patient: {np.mean([len(v) for p, v in patient_visits.items()]).round(2)}", end="\t")
    print(f"#visits/sample: {np.mean([len(v['conditions']) for v in dataset]).round(2)}", end="\t")
    print(f"#conditions/sample(set): {np.mean([len(set([c for visit in v['conditions'] for c in visit])) for v in dataset]).round(2)}", end="\t")
    print(f"#procedures/sample(set): {np.mean([len(set([c for visit in v['procedures'] for c in visit])) for v in dataset]).round(2)}", end="\t")
    print(f"#drugs/sample(set): {np.mean([len(set([c for visit in v['drugs'] for c in visit])) for v in dataset]).round(2)}")
