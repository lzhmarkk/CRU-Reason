TRIPLET_TO_SENTENCE_TEMPLATE = """\
You are an intelligent assistant in the medical field.
Convert all background knowledge into English declarative sentences. Anything you don't think is medically relevant can be deleted.

Background Knowledge:
##TRIPLET_PLACEHOLDER##

--------------------------------
Converted Background Knowledge:
"""

DESCRIPTION_TEMPLATE = f"""\
# Instruction #
Provide a concise and clear one-paragraph description of a medical concept.
Include its key characteristics, clinical relevance, and any important associated information such as symptoms (if applicable), \
usage (for medication/procedure), or related conditions. \
Keep it informative but suitable for a general healthcare audience.

# Example #
Concept: Type 2 Diabetes Mellitus
Description:
Type 2 Diabetes Mellitus is a chronic metabolic disorder characterized by insulin resistance and relative insulin deficiency. \
It commonly occurs in adults and is strongly associated with obesity, sedentary lifestyle, and genetic predisposition. \
Patients often present with hyperglycemia, increased thirst, frequent urination, and fatigue. \
Management typically involves lifestyle modifications, oral hypoglycemic agents, and sometimes insulin therapy.

# Input #
Concept: ##PLACEHOLDER##
Description: [Your output. Nothing else]
"""

TASK_DEFINITION = {
    "mortality": "Mortality: The probability of patient death for a patient's in-hospital visit",
    "readmission": "Readmission: The likelihood of a patient being readmitted to the hospital within 15 days",
    "lenofstay": "Length-of-Stay: The predicted duration a patient will remain in the hospital during an admission",
    "drugrec": "Drug Recommendation: The process of suggesting suitable medications for a patient",
    "diagnosis": "Diagnosis: The forecasting of the presence of specific diseases or conditions"
}
