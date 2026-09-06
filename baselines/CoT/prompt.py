timespans = ["<1 day", "1 day", "2 day", "3 day", "4 day", "5 day", "6 day", "7 day", "1-2 week", ">2 week"]

TASK_MORTALITY = """\
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications. 
Labels: 0 to 9. 9 = mortality, 5 = risky, 0 = survival

Key Considerations:
1. Conditions:
   - Severity of diagnosed conditions (e.g., advanced cancer, severe heart failure, sepsis)
   - Presence of multiple comorbidities
   - Acute vs. chronic nature of conditions
2. Procedures:
   - Invasiveness and complexity of recent procedures 
   - Emergency vs. elective procedures
   - Frequency of life-sustaining procedures (e.g., dialysis, mechanical ventilation)
3. Medications:
   - Use of high-risk medications (e.g., chemotherapy drugs, immunosuppressants)
   - Multiple medication use indicating complex health issues
   - Presence of medications typically used in end-of-life care

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. \
Consider how these factors interact and potentially exacerbate each other. \
Only the patients with extremely very high risk of mortality (definitely die) should be predicted as high scores (7,8,9).
"""

TASK_READMISSION = """\
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge based solely on conditions, procedures, and medications.
Labels: 0 to 9. 9 = readmission within 15 days, 0 = no readmission

Key Considerations:  
1. Conditions:
   - Chronic diseases with high risk of exacerbation (e.g., COPD, heart failure)
   - Conditions requiring close monitoring or frequent adjustments (e.g., diabetes)
   - Recent acute conditions with potential for complications
2. Procedures:
   - Recent major surgeries or interventions with high complication rates
   - Procedures that require extensive follow-up care
   - Incomplete or partially successful procedures
3. Medications:  
   - New medication regimens that may require adjustment
   - Medications with narrow therapeutic windows or high risk of side effects
   - Complex medication schedules that may lead to adherence issues

Note: Analyze the information comprehensively to determine the likelihood of readmission. \
The goal is to accurately distinguish between patients who are likely to be readmitted and those who are not.
"""

TASK_LENOFSTAY = """\
Length of Stay (LOS) Prediction Task:
Objective: Predict the ICU length-of-stay score (0 to 9) for a patient’s hospitalization based solely on conditions and procedures.
Labels: 0 to 9 respectively for each timespan (<1 day, 1 day, 2 day, 3 day, 4 day, 5 day, 6 day, 7 day, 1-2 week, >2 week). \
9 = definite, 5 = moderate possibility, 0 = impossible

# Key Considerations #  
1. Conditions:
   - Severity and stability of acute conditions (e.g. sepsis, respiratory failure)
   - Chronic conditions requiring prolonged monitoring (e.g. advanced heart failure, multi-organ dysfunction)
   - Complications (e.g. infections, delirium) that may delay discharge
2. Procedures:
   - Invasive interventions (e.g. mechanical ventilation, dialysis) typically associated with longer stays
   - Post-procedure recovery complexity (e.g. major surgery vs. minor procedure)
   - Unplanned procedures (e.g. emergency intubation)
"""

TASK_DRUGREC = """\
Drug Recommendation Task:
Score the candidate drug for their suitability in treating a patient's condition based solely on conditions and procedures.
Label: 0 (Unsuitable), 1 (Poor fit), 2 (Possible), 3 (Recommended), 4 (Optimal)

# Key Considerations #
Conditions:
- Target conditions: Specific diseases/indications the drug treats (e.g. diabetes, hypertension, infection type)
- Condition severity: Stage/severity of disease (e.g. mild vs. end-stage)
- Comorbidities: Conditions that may contraindicate drugs (e.g. renal failure, liver disease)
- Acuity: Whether treating acute flare-up or chronic maintenance
Procedures:
- Recent surgeries: Procedures affecting drug metabolism (e.g. GI surgeries impacting absorption)
- Invasive interventions: Lines/ports available for administration (e.g. IV-only drugs if no central line)
- Monitoring capacity: Whether required monitoring is feasible (e.g. frequent lab tests for anticoagulants)
"""

TASK_DIAGNOSIS = """\
Diagnosis Prediction Task:
Score the candidate diseases for their possibility in a patient's subsequent hospital visit based on historical conditions, procedures, and medications.
Label: 0 (Impossible), 1 (Unlikely), 2 (Feasible), 3 (Probable), 4 (Certain)

Key Considerations:
Conditions:
- Undiagnosed symptoms or abnormal findings (e.g. unexplained weight loss, persistent fever)
- Complications/progression of existing chronic diseases (e.g. neuropathic symptoms in diabetic patients)
- Recent acute events or symptom changes (e.g. new-onset chest pain)
Procedures:
- Diagnostic test results (e.g. abnormal imaging findings, biopsy results)
- Surgical findings or intraoperative observations
- Monitoring data patterns (e.g., sustained hypoxemia readings)
Medications:
- Current medications and their therapeutic indications
- Medication side effects mimicking disease symptoms
- Treatment response patterns (e.g. lack of response to antibiotics)
- Medication allergies/adverse reactions
"""

TASK_DESCRIPTION = {"mortality": TASK_MORTALITY, "readmission": TASK_READMISSION,
                    "lenofstay": TASK_LENOFSTAY, "drugrec": TASK_DRUGREC, 'diagnosis': TASK_DIAGNOSIS}

TEMPLATE = f"""\
# Task #
##TASK_PLACEHOLDER##
========================================
# Patient EHR Context #
##CONTEXT_PLACEHOLDER##
##FEWSHOT_PLACEHOLDER##

========================================
Please provide a step-by-step reasoning process based on the patient's context.
The reasoning chain should follow this structured format:
1. Overview: Go over the key information in the patient's EHR context, with the Key Considerations from the task description in mind.
2. Comparison: Analyze the similarities and differences between the patient and similar patients, \
explaining how these factors influence the prediction.
3. Conclusion: Integrate the above information to logically reason towards the predicted outcome.
##INSTRUCTION_PLACEHOLDER##

Important Notes:
- Use the relevant knowledge as needed, but the main focus should be on the patient's EHR context.
- Analyze the similarities and differences between the patient and similar patients to justify the prediction.

========================================
# Output Format #
Reasoning Chain:
1. Overview: [Your Output]
2. Comparison: [Your Output]
3. Conclusion: [Your Output]
##FORMAT_PLACEHOLDER##
"""

TEMPLATE_RAG = f"""\
# Task #
##TASK_PLACEHOLDER##
========================================
# Patient EHR Context #
##CONTEXT_PLACEHOLDER##
##FEWSHOT_PLACEHOLDER##

========================================
# Retrieved Medical Knowledge #
##KNOWLEDGE_PLACEHOLDER##

========================================
Please provide a step-by-step reasoning process based on the patient's context.
The reasoning chain should follow this structured format:
1. Overview: Go over the key information in the patient's EHR context, with the Key Considerations from the task description in mind.
2. Comparison: Analyze the similarities and differences between the patient and similar patients, \
explaining how these factors influence the prediction.
3. Conclusion: Integrate the above information to logically reason towards the predicted outcome.
##INSTRUCTION_PLACEHOLDER##

Important Notes:
- Use the relevant knowledge as needed, but the main focus should be on the patient's EHR context.
- Analyze the similarities and differences between the patient and similar patients to justify the prediction.

========================================
# Output Format #
Reasoning Chain:
1. Overview: [Your Output]
2. Comparison: [Your Output]
3. Conclusion: [Your Output]
##FORMAT_PLACEHOLDER##
"""

OUTPUT_INSTRUCTION = {
    "mortality": "4. Prediction: Summarize the reasoning and state the prediction (0 to 9).",
    "readmission": "4. Prediction: Summarize the reasoning and state the prediction (0 to 9).",
    "lenofstay": "4. Prediction: Summarize the reasoning and state the prediction for each timespan in each line. The output format is:\n" +
                 "\n".join(f"{ts}: 0-9" for ts in timespans),
    "drugrec": "4. Prediction: Score all candidate drugs by their suitability. The output format should be json:\n" +
               '```json\n{\n  "drug_name": {\n    "reason": "your reasoning",\n    "score": [0-4]\n  }\n  ...\n}\n```',
    "diagnosis": "4. Prediction: Score all candidate diseases by their possibility. The output format should be json:\n" +
                 '```json\n{\n  "disease_name": {\n    "reason": "your reasoning",\n    "score": [0-4]\n  }\n  ...\n}\n```'
}
OUTPUT_FORMAT = {
    "mortality": "4. Prediction: [0 to 9]",
    "readmission": "4. Prediction: [0 to 9]",
    "lenofstay": "4. Prediction: [0 to 9 for each timespan]",
    "drugrec": "4. Prediction: [Score drugs in json]",
    "diagnosis": "4. Prediction: [Score diseases in json]"
}
