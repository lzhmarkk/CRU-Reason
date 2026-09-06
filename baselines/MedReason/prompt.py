REFORMAT_TEMPLATE = """\
You are a helpful, pattern-following medical assistant.
Given a patient's context and possible relation paths that link to the task: ##TASK_PLACEHOLDER##.
Select up to ##TOPK_PLACEHOLDER## most correlated entity from the relation paths list based on the context and ##TASK_PLACEHOLDER##.
If total number of paths is less than ##TOPK_PLACEHOLDER##, select all of them.

### Output Format
Strictly follow the JSON structure below.
```json
{
"Paths": [
    {"ranking": 1, "path": "sample_path_1", reason": "reason for choosing this path"},
    .....      
]
}
```

### Input:
Context: 
##CONTEXT_PLACEHOLDER##
Task:
##DESCRIPTION_PLACEHOLDER##
Paths: 
##PATH_PLACEHOLDER##

### Output:
"""
