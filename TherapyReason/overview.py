import os
import json
import time
import random
from pathlib import Path
from modules.kg import graph

from TherapyReason.utils.string import parse_string

template_solution_overview = """\
# Task: Patient Overview
Go over the key information in the patient's context, consider the relationship among conditions and procedures in the context. Summarize the patient's context in one paragraph briefly.

## Patient's Context
{solution_string}

## Output Format
Patient Overview: [Your output]
"""


def solution2string(solution):
    solution_name, solution_relations = "", ""

    context = solution['conditions'] + solution['procedures'] + solution['drugs']
    if solution['conditions']:
        solution_name += "### Conditions\n" + "".join(x + "\n" for x in solution['conditions'])
    if solution['procedures']:
        solution_name += "### Procedures\n" + "".join(x + "\n" for x in solution['procedures'])
    if solution['drugs']:
        solution_name += "### Medications\n" + "".join(x + "\n" for x in solution['drugs'])

    solution_name += f"### Relations among Concepts\n"
    for node, neighbor in solution['subgraph'].items():
        for edge, dst in neighbor:
            src = graph.id2entity[int(node)].lower()
            rel = graph.id2relation[int(edge)].replace('_', '').lower()
            dst = graph.id2entity[int(dst)].lower()
            if src in context and dst in context:
                solution_relations += f"{src}->{rel}->{dst}\n"

    return solution_name + solution_relations


def post_process_solution_overview(prob, response, output):
    overview = response.replace("*", "").split("Patient Overview")[-1]
    return prob, parse_string(overview), output


def process_solution_overview_wrap(cache_dir, llm):
    def process_solution_overview(key, item, **kwargs):
        cache_file = Path(os.path.join(cache_dir, f"{key}.json"))
        os.makedirs(cache_file.parent, exist_ok=True)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            time.sleep(random.randint(1, 10))
            solution_obj = item

            cached = []
            for visit_id, visit in enumerate(solution_obj['solutions']):
                cached_visit = []
                for solution_id, solution in enumerate(visit):
                    query = template_solution_overview.format(solution_string=solution2string(solution))
                    _, overview, _, usage = llm.understand(query, post_process=post_process_solution_overview, **kwargs)
                    cached_visit.append({"query": query, "overview": overview, "usage": usage.dict()})

                cached.append(cached_visit)
            cache_file.write_text(json.dumps(cached, indent=4), encoding="utf-8")

        return key, cached

    return process_solution_overview
