from data.dataloader import load_mappings
from modules.kg import graph

MOTIF_RULE0 = [
    "C->C",
    "C->P",
    "C->D",
    "D->P",
    "P->D"
]
MOTIF_RULE1 = [
    "C->C",
    "C->P",
    "C->D",
    "D->P",
    "P->D",
    "P->C",
    "D->C"
]
MOTIF_RULE2 = [
    "C->C",
    "C->P",
    "C->D",
    "D->P",
    "P->D",
    "P->C",
    "D->C",
    "P->P",
    "D->D",
]

condition_dict, procedure_dict, drug_dict = load_mappings(inv_mapping=False)
is_C = lambda c: c in condition_dict.values() and c not in procedure_dict.values() and c not in drug_dict.values()
is_P = lambda c: c in procedure_dict.values() and c not in condition_dict.values() and c not in drug_dict.values()
is_D = lambda c: c in drug_dict.values() and c not in condition_dict.values() and c not in procedure_dict.values()
get_type = lambda c: "C" if is_C(c) else ("P" if is_P(c) else ("D" if is_D(c) else "E"))


def get_motif(src, edge, dst):
    src_name = graph.id2entity[src].lower()
    edge = graph.graph.relation[edge].item()
    dst_name = graph.id2entity[dst].lower()
    motif = f"{get_type(src_name)}->{get_type(dst_name)}"
    return motif


def check_motif(src, edge, dst, rules):
    motif = get_motif(src, edge, dst)
    return motif in rules
