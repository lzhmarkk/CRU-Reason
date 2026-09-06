import json
from pathlib import Path
from modules.motif import get_motif, check_motif, MOTIF_RULE0, MOTIF_RULE1, MOTIF_RULE2
from modules.blacklists import *
from modules.kg import graph
from utils.llm import LLM


def llm_relevance(context,
                  candidate, relation, source,
                  direction,
                  llm: LLM):
    candidate = graph.id2entity[candidate]
    relation = graph.id2relation[relation]
    source = graph.id2entity[source]
    context = ", ".join(graph.id2entity[node] for node in context)

    if direction == 'forward':
        triplet = f"{source} -> {relation} -> {candidate}"
    elif direction == 'backward':
        triplet = f"{candidate} -> {relation} -> {source}"
    else:
        raise ValueError('direction must be either forward or backward')

    prompt = f"""\
# Instruction
Determine whether the given medical concept is relevant to the provided context (a set of other medical concepts). \
Your task is to determine whether the concept and at least one concept in the context are clinically relevant, \
i.e. related to the same disease, syndrome, or body part.
Briefly justify your reasoning in one sentence. \
Then, provide your prediction with "YES" if the node is relevant to at least one concept in the context, otherwise "NO".

# Example
Concept: Hypertension
Context: Diabetes, Coronary artery disease, Retinopathy
Relation: Coronary artery disease -> can occur with -> Hypertension
Answer: YES. Hypertension and coronary artery disease are both cardiovascular conditions, affecting the same body system (heart and blood vessels).

Concept: Gastritis
Context: Peptic Ulcer, GERD, Colitis
Relation: Peptic Ulcer -> increase risk of -> Gastritis
Answer: YES. Gastritis, peptic ulcer, and GERD all involve the stomach (upper GI tract), whereas colitis affects the colon (not directly relevant).

Concept: Osteoarthritis
Context: Asthma, Eczema, Allergic Rhinitis
Relation: Allergic Rhinitis -> can coexist with -> Osteoarthritis
Answer: NO. Osteoarthritis is a joint disorder, while the context conditions are all respiratory/allergic and unrelated anatomically or pathologically.

# Input
Concept: {candidate}
Context: {context}
Relation: {triplet}

# Output
Answer: [YES or NO]
"""

    def post_process(prob, response, output):
        answer = response.split("Answer")[-1]
        if "YES" in answer and "NO" not in answer:
            prediction = True
        elif "NO" in answer and "YES" not in answer:
            prediction = False
        else:
            raise ValueError

        # print(triplet)
        # print(response.replace("\n\n", "\n"), end="\n\n\n")
        return prob, prediction, output

    _, prediction, _ = llm.understand(prompt, post_process=post_process)
    return prediction


def semantic_relevance(source, relation, candidate):
    cached_relevance = json.load(Path("TherapyReason/cache/llm_pairwise.json").open('r', encoding='utf-8'))

    candidate = graph.id2entity[candidate]
    source = graph.id2entity[source]

    for key in (f"{source}-{candidate}", f"{candidate}-{source}"):
        if key in cached_relevance:
            return bool(cached_relevance[key])
    return None


def embedding_relevance(source, relation, candidate, threshold=0.4):
    cached_relevance = json.load(Path("TherapyReason/cache/emb_pairwise.json").open('r', encoding='utf-8'))

    candidate = graph.id2entity[candidate]
    source = graph.id2entity[source]

    for key in (f"{source}-{candidate}", f"{candidate}-{source}"):
        if key in cached_relevance:
            return cached_relevance[key] >= threshold
    return False


def is_relevant_edge(src: int, edge: int, dst: int,
                     direction='forward',
                     existing_nodes=None,
                     included_nodes=None,
                     motifs=None,
                     use_node_rule=True,
                     use_edge_rule=True,
                     llm=None,
                     use_semantic=False,
                     use_embedding=False,
                     ):
    """
    whether a candidate edge is relevant and can be included in BFS
    Args:
        src (int): source node id
        edge (int): candidate edge id
        dst (int): candidate neighbor id
        direction (str): forward: (src)->[edge]->(dst); backward: (dst)->[edge]->(src)
        existing_nodes (list|set): existing nodes within the subgraph(tree) during BFS
        included_nodes (int): nodes that we care about. Nodes not in `included_nodes` will be discarded
        motifs (list): edge motifs that we care about. Edges not in `motifs` will be discarded
        use_node_rule (bool): whether we should use node blacklists/whitelists or not
        use_edge_rule (bool): whether we should use edge blacklists/whitelists or not
        llm (LLM): if given, LLM-based reasoning will be applied
    Returns:
        (bool): whether this edge is relevant and should be included in BFS
    """
    if included_nodes is not None and isinstance(included_nodes, list):
        if dst not in included_nodes:
            return False

    if motifs is not None:
        assert motifs in [MOTIF_RULE0, MOTIF_RULE1, MOTIF_RULE2]
        if not check_motif(src, edge, dst, rules=motifs):
            return False

    if use_node_rule:
        if not node_rule_filtering(src, edge, dst):
            return False

    # negative and vague ara always banned
    relation_name = graph.id2relation[edge].lower()
    if any(w in relation_name.lower() for w in negative_relation + vague_relation):
        return False

    # For C->C, order: semantic similarity -> embedding similarity -> rule similarity
    # For other, use rule only
    if use_semantic and get_motif(src, edge, dst) == "C->C":
        semantic_rel = semantic_relevance(src, edge, dst)
        if not semantic_rel:
            return False
    elif use_embedding and get_motif(src, edge, dst) == "C->C":
        embedding_rel = embedding_relevance(src, edge, dst, threshold=0.4)
        if not embedding_rel:
            return False
    elif use_edge_rule:
        if any(w in relation_name.lower() for w in weak_relation + blacklist_relation):
            return False

    if llm is not None and isinstance(llm, LLM) and len(existing_nodes) > 0:
        if not llm_relevance(context=existing_nodes,
                             candidate=dst, relation=edge, source=src,
                             direction=direction,
                             llm=llm):
            return False

    return True
