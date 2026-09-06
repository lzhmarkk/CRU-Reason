TEMPLATE_KEYWORD_SYSTEM = '''
You are a helpful medical expert with extensive medical knowledge. \
You should summarize the context concisely into several keywords, ensuring that all the essential information is covered comprehensively. \
Your summary should be informative and beneficial for predicting in-hospital mortality.
'''

TEMPLATE_KEYWORD_USER = '''
Here is the context:
##CONTEXT_PLACEHOLDER##

Multiple search keywords are allowed as a list of strings and format that in the keywords field as {"keywords": List(Your search queries)} \
which can be loaded by python json library. Your summary should be informative and beneficial for {task} task.

Here is an example of the format you should output:
```json
{"keywords": ["End-Stage Renal Disease", "Hypertensive kidney damage", "Low Albumin level", "Carbon dioxide binding power", \
"Low Diastolic blood pressure", "Low Blood chlorine"]}
```

Please respond with only the information you want to search in JSON format without any additional information:
'''
