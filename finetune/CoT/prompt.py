from baselines.CoT.prompt import timespans

TASK_MORTALITY_SHORT = """\
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications. 
Labels: 0 to 9. 9 = mortality, 5 = risky, 0 = survival
"""

TASK_READMISSION_SHORT = """\
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge based solely on conditions, procedures, and medications.
Labels: 0 to 9. 9 = readmission within 15 days, 0 = no readmission
"""

TASK_LENOFSTAY_SHORT = """\
Length of Stay (LOS) Prediction Task:
Objective: Predict the ICU length-of-stay score (0 to 9) for a patient’s hospitalization based solely on conditions and procedures.
Labels: 0 to 9 respectively for each timespan (<1 day, 1 day, 2 day, 3 day, 4 day, 5 day, 6 day, 7 day, 1-2 week, >2 week). \
9 = definite, 5 = moderate possibility, 0 = impossible
"""

TASK_DRUGREC_SHORT = """\
Drug Recommendation Task:
Score the candidate drug for their suitability in treating a patient's condition based solely on conditions and procedures.
Label: 0 (Unsuitable), 1 (Poor fit), 2 (Possible), 3 (Recommended), 4 (Optimal)
"""

TASK_DIAGNOSIS_SHORT = """\
Diagnosis Prediction Task:
Objective: Predict potential diagnosis conditions based on the patient's historical conditions, procedures, and medications.
Label: 0 (Impossible), 1 (Unlikely), 2 (Feasible), 3 (Probable), 4 (Certain)
"""

TASK_DESCRIPTION_SHORT = {"mortality": TASK_MORTALITY_SHORT, "readmission": TASK_READMISSION_SHORT,
                          "lenofstay": TASK_LENOFSTAY_SHORT, "drugrec": TASK_DRUGREC_SHORT,
                          'diagnosis': TASK_DIAGNOSIS_SHORT}

FINETUNE_TEMPLATE = f"""\
# Task #
##TASK_PLACEHOLDER##
========================================
# Patient EHR Context #
##CONTEXT_PLACEHOLDER##
##FEWSHOT_PLACEHOLDER##

========================================
# Ground Truth Prediction #
##TRUTH_PLACEHOLDER##

========================================
Please provide a step-by-step reasoning process that leads to the ground truth based on the patient's context.
The reasoning chain should follow this structured format:
1. Overview: Go over the key information in the patient's EHR context, with the Key Considerations from the task description in mind.
2. Comparison: Analyze the similarities and differences between the patient and similar patients, \
explaining how these factors influence the prediction.
3. Reasoning: Integrate the above information to logically reason towards the ground truths.
##INSTRUCTION_PLACEHOLDER##

Important Notes:
- Do not mention the ground truth label in the reasoning process.
- Use the relevant knowledge as needed, but the main focus should be on the patient's EHR context.
- Analyze the similarities and differences between the patient and similar patients to justify the prediction.

After generating the reasoning chain, please review it and indicate your confidence in the reasoning chain at the end.
Options of confidence: [Very Confident, Confident, Neutral, Not Confident, Very Not Confident.]

========================================
# Output Format #
Reasoning Chain:
1. Overview: [Your Output]
2. Comparison: [Your Output]
3. Reasoning: [Your Output]
##FORMAT_PLACEHOLDER##

# Confidence #
[choose one: "Very Confident", "Confident", "Neutral", "Not Confident", "Very Not Confident"]
"""

FINETUNE_OUTPUT_INSTRUCTION = {
    "mortality": "4. Conclusion: Summarize the reasoning without mentioning the ground truth label.",
    "readmission": "4. Conclusion: Summarize the reasoning without mentioning the ground truth label.",
    "lenofstay": "4. Conclusion: Summarize the reasoning without mentioning the ground truth label. " +
                 "The output format is:\n" + "\n".join(f"{ts}: 0-9" for ts in timespans),
    "drugrec": "4. Conclusion: Summarize candidate drugs about their suitability without mentioning the ground truth label. " +
               "The output format should be json:\n" +
               '```json\n{\n  "drug_name": {\n    "score": [0-4. copy the above score],\n    "reason": "your reasoning"\n  }\n  ...\n}\n```',
    "diagnosis": "4. Conclusion: Summarize candidate diseases about their possibility without mentioning the ground truth label. " +
                 "The output format should be json:\n" +
                 '```json\n{\n  "disease_name": {\n    "score": [0-4. copy the above score],\n    "reason": "your reasoning"\n  }\n  ...\n}\n```',
}
FINETUNE_OUTPUT_FORMAT = {
    "mortality": "4. Conclusion: [0 to 9]",
    "readmission": "4. Conclusion: [0 to 9]",
    "lenofstay": "4. Conclusion: [0 to 9 for each timespan]",
    "drugrec": "4. Conclusion: [Score each drug in json]",
    "diagnosis": "4. Conclusion: [Score each disease in json]",
}
