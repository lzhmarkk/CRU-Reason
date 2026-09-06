import re
from utils.llm import LLM


class Doctor:
    def __init__(self, chat_model: LLM, task: str):
        self.chat_model: LLM = chat_model
        self.task = task
        assert self.task in ['drugrec', 'diagnosis']
        self.target_name = {"drugrec": "medication", "diagnosis": "disease"}[self.task]

    def general_info_summary(self, patient):
        chief_complaint = None
        current_medical_history = "\n".join(patient['conditions'][-1])
        disease_history = ""
        for visit_id, visit in enumerate(patient['conditions'][:-1]):
            disease_history += f"Visit {visit_id + 1}: "
            disease_history += "[" + ", ".join(visit) + "]\n"
        if not disease_history:
            disease_history = "No history"

        # [Chief Complaint]: {chief_complaint}
        prompt = f"""\
You are an excellent AI medical expert, capable of summarizing key diagnostic information based on a patient's medical records. \
Below is a partial excerpt from a real patient's electronic health record. \
Please carefully review the following details to understand the patient's condition.
[Present Complaint]:
{current_medical_history}
[Past Medical History]:
{disease_history}

# Task
Based on the above content, summarize the key information relevant to diagnosis and treatment, and generate a structured summary report \
with specific formatting requirements.
Report Format Requirements: Please follow the format below, filling in the "[]" sections. Use concise language wherever possible.

# Output Format
1. Presenting Symptoms: []
2. Recent Medical Visits: [] (Write None if not applicable)
3. Past Medical Conditions: [] (Write None if not applicable)
4. Past Surgical History: [] (Write None if not applicable)
5. Medication Usage: [] (Write None if not applicable)
6. Summary:
"""

        def post_process_info(prob, response, output):
            return prob, (response, response.split("Summary")[-1].strip(" :[]")), output

        _, (response, summary), _, usage = self.chat_model.understand(query=prompt, post_process=post_process_info)
        return prompt, summary, usage

    def examination_summary(self, patient):
        body_check = "\n".join(patient['procedures'][-1])
        auxiliary_exam = None

        # [Auxiliary Test Results]: {auxiliary_exam}
        prompt = f"""\
You are an excellent AI medical expert, capable of summarizing key diagnostic information based on a patient’s test results. \
The patient's test results are shown below:"
[Physical Examination Findings]:
{body_check}  

Currently, the patient's test results contain a lot of redundant information. Please summarize the above test results.

# Task
Please summarize and condense the patient's test results above, retaining information useful for diagnosis and removing content \
of little diagnostic significance.

# Output Requirements
Please follow the format below, filling in the "[]" sections. Use concise language wherever possible.
1. Physical Examination Findings: []
2. Summary:
"""

        def post_process_exam(prob, response, output):
            return prob, (response, response.split("Summary")[-1].strip(" :[]")), output

        _, (response, summary), _, usage = self.chat_model.understand(query=prompt, post_process=post_process_exam)
        return prompt, summary, usage

    def direct_reason(self, fst_rd_summary, scd_rd_summary, topn=50):
        def direct_reason_post_process(prob, response, output):
            prediction = []
            lines = response.splitlines()
            for line in lines:
                if line.strip('*# ').startswith('- Prediction'):
                    prediction.append(line.split(":")[-1].strip())

            return prob, (response, prediction), output

        prompt = f"""\
You are an excellent AI medical expert capable of diagnosing diseases based on a patient's condition.

# Patient's Basic Information
{fst_rd_summary}

# Patient's Test Results
{scd_rd_summary}

# Task Instructions
Based on the patient's symptoms, medical history, past conditions, and test results, predict the possible {self.task} the patient may have \
(you may provide the top-{topn} most likely prediction). Format your response as specified below.
Output only the predicted results — do not include any additional content.

# Format Requirements
"""
        for i in range(topn):
            prompt += f"- Prediction {i + 1}:\n"

        _, (response, direct_reasoning), _, usage = self.chat_model.understand(prompt, post_process=direct_reason_post_process)
        return prompt, response, direct_reasoning, usage

    def analysis(self, candidate, formatted_knowledge, dis_history, auxiliary_exam):
        def post_process_analysis(prob, response, output):
            prediction = response.split("be confirmed as prediction")[-1].strip()
            score = re.findall(r"(\d)", prediction)[0]
            return prob, (response, int(score)), output

        analysis_template = f"""\
Evaluation of the likelihood of candidate {self.target_name} *{candidate}*:
1. Relevance score to patient's medical history: [?] (out of 10, score only, no explain)
2. Relevance score to patient's procedure history: [?] (out of 10, score only, no explain)
3. Are there any errors or misleading information in the provided knowledge? [?]
4. Can this {self.target_name} be confirmed as prediction: [?] \
(Choose from 0 (Unsuitable), 1 (Poor fit), 2 (Possible), 3 (Recommended), 4 (Optimal), no explain)
"""

        prompt = f"""\
You are an excellent AI medical expert. You can analyze the rationality of existing {self.target_name} based on patient symptoms, medical history, \
and test results. Please carefully review the following patient information.
Patient's medical history: {dis_history}
Patient's procedure history: {auxiliary_exam}
A physician's preliminary {self.target_name} for this patient: {candidate}.

You need to evaluate the validity of this {self.target_name}. We've provided you with relevant information from a medical knowledge graph:
{formatted_knowledge}

# Task
Please evaluate this {self.target_name} based on the knowledge graph information we have provided

# Format requirements
{analysis_template}

# Your analysis, as brief as possible
"""

        _, (response, score), _, usage = self.chat_model.understand(prompt, post_process=post_process_analysis)
        return prompt, response, score, usage
