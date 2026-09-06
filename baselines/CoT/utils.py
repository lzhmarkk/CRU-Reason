from baselines.CoT.prompt import timespans


def get_similar_patient_label(dataset, task):
    patient_labels = {}
    for patient in dataset:
        sample_id = patient['patient_id'] + "_" + patient['visit_id']
        if task in ['mortality', 'readmission']:
            patient_labels[sample_id] = patient['label'] * 4 + 3
        elif task in ['lenofstay']:
            patient_labels[sample_id] = timespans[patient['label']]
        elif task in ['drugrec']:
            patient_labels[sample_id] = "\n".join(patient['drugs'])
        elif task in ['diagnosis']:
            patient_labels[sample_id] = "\n".join(patient['label'])
        else:
            raise ValueError()

    return patient_labels
