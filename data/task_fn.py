from pyhealth.data import Patient, Visit


def load_visit(visit: Visit, dataset):
    if dataset == 'mimic3':
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
    elif dataset == 'mimic4':
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
    else:
        raise ValueError()

    return conditions, procedures, drugs


def remap_visit(conditions, procedures, drugs, **kwargs):
    return [kwargs['condition_dict'][item] for item in conditions], \
        [kwargs['procedure_dict'][item] for item in procedures], \
        [kwargs['drug_dict'][item] for item in drugs]


def drug_recommendation_fn(patient, dataset, **kwargs):
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions, procedures, drugs = load_visit(visit, dataset)
        drugs = [drug[:4] for drug in drugs] if dataset == 'mimic4' else drugs
        conditions, procedures, drugs = remap_visit(conditions, procedures, drugs, **kwargs)

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs_all": [drugs],
                "drugs": drugs  # for drug rec label
            }
        )

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + samples[i]["drugs_all"]
    return samples


def mortality_prediction_fn(patient, dataset, **kwargs):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions, procedures, drugs = load_visit(visit, dataset)
        conditions, procedures, drugs = remap_visit(conditions, procedures, drugs, **kwargs)

        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label
            }
        )

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + samples[i]["drugs"]
    return samples


def readmission_prediction_fn(patient: Patient, dataset, time_window=15, **kwargs):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions, procedures, drugs = load_visit(visit, dataset)
        conditions, procedures, drugs = remap_visit(conditions, procedures, drugs, **kwargs)

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": readmission_label
            }
        )

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + samples[i]["drugs"]
    return samples


def length_of_stay_prediction_fn(patient: Patient, dataset, **kwargs):
    samples = []

    for visit in patient:
        conditions, procedures, drugs = load_visit(visit, dataset)
        conditions, procedures, drugs = remap_visit(conditions, procedures, drugs, **kwargs)

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": los_category
            }
        )

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + samples[i]["drugs"]
    return samples


def diagnosis_prediction_fn(patient, dataset, **kwargs):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        conditions, procedures, drugs = load_visit(visit, dataset)
        conditions, procedures, drugs = remap_visit(conditions, procedures, drugs, **kwargs)

        next_conditions, next_procedures, next_drugs = load_visit(next_visit, dataset)
        next_conditions, next_procedures, next_drugs = remap_visit(next_conditions, next_procedures, next_drugs, **kwargs)

        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0 or len(next_conditions) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": next_conditions
            }
        )

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + samples[i]["drugs"]
    return samples


def categorize_los(days: int):
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9
