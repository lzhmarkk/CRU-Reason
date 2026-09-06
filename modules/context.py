import re
import itertools


def get_concepts(sample, task, with_history=False, flatten=False):
    include_drug = task in ['mortality', 'readmission', 'diagnosis']

    conditions = sample['conditions']
    procedures = sample['procedures']
    drugs = sample['drugs'] if include_drug else [[]] * len(conditions)

    if not with_history:
        conditions = conditions[-1:]
        procedures = procedures[-1:]
        drugs = drugs[-1:]

    if flatten:
        conditions = list(set(itertools.chain(*conditions)))
        procedures = list(set(itertools.chain(*procedures)))
        drugs = list(set(itertools.chain(*drugs)))

    return conditions, procedures, drugs
