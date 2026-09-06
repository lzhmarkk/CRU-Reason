import copy
import os
import csv
import pickle
import random
from functools import partial
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, SampleEHRDataset
from data.task_fn import drug_recommendation_fn, mortality_prediction_fn, readmission_prediction_fn, \
    length_of_stay_prediction_fn, diagnosis_prediction_fn
from pyhealth.datasets import get_dataloader, split_by_patient
from data.utils import print_statistics


def load_mappings(inv_mapping=False):
    condition_mapping_file = "./data/resources/CCSCM.csv"
    procedure_mapping_file = "./data/resources/CCSPROC.csv"
    drug_file = "./data/resources/ATC.csv"

    condition_dict, condition_dict_inv = {}, {}
    with open(condition_mapping_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()
            condition_dict_inv[row['name'].lower()] = row['code']

    procedure_dict, procedure_dict_inv = {}, {}
    with open(procedure_mapping_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            procedure_dict[row['code']] = row['name'].lower()
            procedure_dict_inv[row['name'].lower()] = row['code']

    drug_dict, drug_dict_inv = {}, {}
    with open(drug_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name'].lower()
                drug_dict_inv[row['name'].lower()] = row['code']

    if not inv_mapping:
        return condition_dict, procedure_dict, drug_dict
    else:
        return condition_dict, procedure_dict, drug_dict, condition_dict_inv, procedure_dict_inv, drug_dict_inv


def load_raw_dataset(dataset, task):
    # assign task & remap
    dataset_task_path = f"./data/dataset/{dataset}_{task}.pkl"
    if os.path.exists(dataset_task_path):
        with open(dataset_task_path, "rb") as f:
            full_dataset = pickle.load(f)
    else:
        dataset_path = f"./data/dataset/{dataset}.pkl"

        if os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                ds = pickle.load(f)
        else:
            if dataset == "mimic3":
                ds = MIMIC3Dataset(root="./data/original_dataset/mimiciii/1.4/",
                                   tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                                   code_mapping={
                                       "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                                       "ICD9CM": "CCSCM",
                                       "ICD9PROC": "CCSPROC"
                                   })
            elif dataset == "mimic4":
                ds = MIMIC4Dataset(root="./data/original_dataset/mimiciv/2.0/hosp/",
                                   tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                                   code_mapping={
                                       "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                                       "ICD9CM": "CCSCM",
                                       "ICD9PROC": "CCSPROC",
                                       "ICD10CM": "CCSCM",
                                       "ICD10PROC": "CCSPROC",
                                   },
                                   dev=False)
            else:
                raise ValueError()

            os.makedirs("./data/dataset", exist_ok=True)
            with open(dataset_path, "wb") as f:
                pickle.dump(ds, f)

        condition_dict, procedure_dict, drug_dict = load_mappings()
        kwargs = {"condition_dict": condition_dict, "procedure_dict": procedure_dict, "drug_dict": drug_dict,
                  "dataset": dataset}
        if task == "drugrec":
            full_dataset = ds.set_task(partial(drug_recommendation_fn, **kwargs), task_name=task)
        elif task == "mortality":
            full_dataset = ds.set_task(partial(mortality_prediction_fn, **kwargs), task_name=task)
        elif task == "readmission":
            full_dataset = ds.set_task(partial(readmission_prediction_fn, **kwargs), task_name=task)
        elif task == "lenofstay":
            full_dataset = ds.set_task(partial(length_of_stay_prediction_fn, **kwargs), task_name=task)
        elif task == 'diagnosis':
            full_dataset = ds.set_task(partial(diagnosis_prediction_fn, **kwargs), task_name=task)
        else:
            raise ValueError()

        with open(dataset_task_path, "wb") as f:
            pickle.dump(full_dataset, f)

    if dataset == 'mimic4':
        sample_dataset = mimic4_downsample(full_dataset.samples, task)
    else:
        sample_dataset = full_dataset

    return full_dataset, sample_dataset


def mimic4_downsample(sample_dataset: list, task: str):
    random.seed(42)
    sample_dataset = copy.deepcopy(sample_dataset)

    if task in ['mortality', 'readmission']:
        random.shuffle(sample_dataset)  # KARE forgets to shuffle

        # identical sample orders to KARE
        # sample_dataset_dict = {f"{s['patient_id']}_{s['visit_id']}": s for s in sample_dataset}
        # sample_dataset = [sample_dataset_dict[k] for k in list(sample_dataset_dict.keys())]

        sample_dataset_new = []
        patient_set = set()
        for sample in sample_dataset:
            if sample['label'] == 1 and sample['patient_id'] not in patient_set:
                if len(sample['conditions']) <= 10:  # KARE has bug here
                    sample_dataset_new.append(sample)
                    patient_set.add(sample['patient_id'])
                if len(sample_dataset_new) == 5000:
                    break

        for sample in sample_dataset:
            if sample['label'] == 0 and sample['patient_id'] not in patient_set:
                if len(sample['conditions']) <= 10:
                    sample_dataset_new.append(sample)
                    patient_set.add(sample['patient_id'])
                if len(sample_dataset_new) == 10000:
                    break

    elif task in ['lenofstay', 'drugrec']:
        sample_dataset_new = list(filter(lambda sample: len(sample['conditions']) <= 10, sample_dataset))
        random.shuffle(sample_dataset_new)
        sample_dataset_new = sample_dataset_new[:40000]  # similar scale with mimic3

    elif task in ['diagnosis']:
        sample_dataset_new = list(filter(lambda sample: len(sample['conditions']) <= 10, sample_dataset))
        random.shuffle(sample_dataset_new)
        sample_dataset_new = sample_dataset_new[:10000]  # similar scale with mimic3

    else:
        raise ValueError()

    return SampleEHRDataset(sample_dataset_new)


def load_dataset(dataset, task, ratios=None, train_ratio=1.0, n_train_samples=-1, n_val_samples=-1, n_test_samples=1000):
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be in (0, 1]")
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]

    print(f"Dataset: {dataset}, Task: {task}")
    print("Loading dataset...")

    full_dataset, sample_dataset = load_raw_dataset(dataset, task)

    # split train/val/test
    print(f"Splitting dataset by {ratios}, training ratio: {train_ratio}")
    train_dataset, val_dataset, test_dataset = split_by_patient(full_dataset, ratios, seed=528)
    train_patients = set([item['patient_id'] for item in train_dataset])
    val_patients = set([item['patient_id'] for item in val_dataset])
    test_patients = set([item['patient_id'] for item in test_dataset])
    all_train_patients = train_patients.copy()
    if train_ratio < 1.0:
        patient_ids = sorted(train_patients)
        n_patients = int(len(patient_ids) * train_ratio)
        train_patients = set(random.Random(528).sample(patient_ids, n_patients))

    datasets = {"train": [], "val": [], "test": []}
    for sample in sample_dataset:
        if sample['patient_id'] in train_patients:
            datasets['train'].append(sample)
        elif sample['patient_id'] in val_patients:
            datasets['val'].append(sample)
        elif sample['patient_id'] in test_patients:
            datasets['test'].append(sample)
        elif sample['patient_id'] in all_train_patients:
            continue  # deliberately excluded by train_ratio
        else:
            raise ValueError()

    # select samples
    if n_train_samples > 0:
        datasets['train'] = datasets['train'][:n_train_samples]
    if n_val_samples > 0:
        datasets['val'] = datasets['val'][:n_val_samples]
    if n_test_samples > 0:
        datasets['test'] = datasets['test'][:n_test_samples]

    return datasets, full_dataset


def load_dataloader(dataset, task, batch_size, ratios=None, train_ratio=1.0,
                    n_train_samples=-1, n_val_samples=-1, n_test_samples=1000):
    datasets, full_dataset = load_dataset(dataset, task, ratios, train_ratio,
                                          n_train_samples, n_val_samples, n_test_samples)

    # features
    features = {"max_visits": max(len(s['conditions']) for s in full_dataset)}

    # statistics
    print_statistics(datasets['train'])
    print_statistics(datasets['val'])
    print_statistics(datasets['test'])

    train_loader = get_dataloader(datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(datasets['val'], batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(datasets['test'], batch_size=batch_size, shuffle=False)
    return full_dataset, {"train": train_loader, "val": val_loader, "test": test_loader}, features


# preprocess
if __name__ == "__main__":
    datasets = [
        "mimic3",
        "mimic4"
    ]
    tasks = [
        "drugrec",
        "mortality",
        "readmission",
        "lenofstay",
        "diagnosis"
    ]

    for dataset in datasets:
        for task in tasks:
            load_dataset(dataset, task)
