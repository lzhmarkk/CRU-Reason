KARE_FINETUNE_TEMPLATE = f"""\
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
# Ground Truth Prediction #
##TRUTH_PLACEHOLDER##

========================================
Please provide a step-by-step reasoning process that leads to the ground truth based on the patient's context.
The reasoning chain should follow this structured format:
1. Overview: Go over the key information in the patient's EHR context, with the Key Considerations from the task description in mind.
2. Knowledge: Highlight the retrieved medical knowledge pertinent to the patient's condition.
3. Comparison: Analyze the similarities and differences between the patient and similar patients, \
explaining how these factors influence the prediction.
4. Reasoning: Integrate the above information to logically reason towards the ground truths.
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
2. Knowledge: [Your Output]
3. Comparison: [Your Output]
4. Reasoning: [Your Output]
##FORMAT_PLACEHOLDER##

# Confidence #
[choose one: "Very Confident", "Confident", "Neutral", "Not Confident", "Very Not Confident"]
"""
