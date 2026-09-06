KARE_TEMPLATE = f"""\
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
1. Patient Overview: Go over the key information in the patient's EHR context, with the Key Considerations from the task description in mind.
2. Relevant Retrieved Medical Knowledge: Highlight the retrieved medical knowledge pertinent to the patient's condition.
3. Comparison with Similar Patients: Analyze the similarities and differences between the patient and similar patients, explaining how these factors influence the prediction.
4. Conclusion Towards Prediction: Integrate the above information to logically reason towards the predicted outcome.
##INSTRUCTION_PLACEHOLDER##

Important Notes:
- Use the relevant knowledge as needed, but the main focus should be on the patient's EHR context.
- Analyze the similarities and differences between the patient and similar patients to justify the prediction.

========================================
# Output Format #
Reasoning Chain:
1. Overview: [Your Output]
2. Relevant Knowledge: [Your Output]
3. Comparison: [Your Output]
4. Conclusion: [Your Output]
##FORMAT_PLACEHOLDER##
"""
