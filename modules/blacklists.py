from data.dataloader import load_mappings
from modules.kg import graph

blacklist_entity = ['residual', 'e codes:']
negative_relation = ['not', "doesn't", "didn't", "isn't", "aren't", "rare", "seldom", "never",
                     "unlike", "without", "more", "less", "difficult", "hard", "absence", "limited"]
vague_relation = ['if', "uncertain", "unclear", "uncontrolled", "unrelated", "vague", 'involv', 'includ', 'when', "may", "might", "after"]
weak_relation = ["serious", "challeng", "comorbid", "interact", 'relate', 'affect', "couple", "common", "indicate",
                 "associat", "complicat", "coexist", 'combine', "alongside", "accompan", "context", "influence", "complex"]
blacklist_relation = ['require', "crucial", "severe", "manage", "occur", "risk", "compromise", "lead", "cause", "impair", "impact"]

condition_dict, procedure_dict, drug_dict = load_mappings(inv_mapping=False)
is_C = lambda c: c in condition_dict.values() and c not in procedure_dict.values() and c not in drug_dict.values()
is_P = lambda c: c in procedure_dict.values() and c not in condition_dict.values() and c not in drug_dict.values()
is_D = lambda c: c in drug_dict.values() and c not in condition_dict.values() and c not in procedure_dict.values()
get_type = lambda c: "C" if is_C(c) else ("P" if is_P(c) else ("D" if is_D(c) else "E"))


def node_rule_filtering(src, edge, dst):
    src_name = graph.id2entity[src].lower() if isinstance(src, int) else str(src).lower()
    dst_name = graph.id2entity[dst].lower() if isinstance(dst, int) else str(dst).lower()
    if any(w in src_name.lower() or w in dst_name.lower() for w in blacklist_entity):
        return False

    return True


def edge_rule_filtering(src, edge, dst):
    relation_name = graph.id2relation[edge].lower() if isinstance(edge, int) else str(edge).lower()
    if any(w in relation_name.lower() for w in negative_relation + vague_relation + weak_relation + blacklist_relation):
        return False

    return True
