import numpy as np
from collections import deque
from utils.llm import LocalModel
from modules.motif import is_C, is_P, is_D
from modules.kg import graph, adjacency_list, adjacency_list_inv
from modules.blacklists import node_rule_filtering, edge_rule_filtering
from TherapyReason.utils.subgraph import get_subgraph, print_subgraph, get_nodes


def get_subgraph_connectivity(subgraph, other, method, **kwargs):
    conditions_in_subgraph = [node for node in subgraph.keys() if is_C(graph.id2entity[node])]
    conditions_in_other = [node for node in other.keys() if is_C(graph.id2entity[node])]

    if method == 'share':
        shared_node = set(conditions_in_subgraph).intersection(set(conditions_in_other))
        if len(shared_node) > 0:
            # print("Shared nodes:", [graph.id2entity[node] for node in shared_node])
            return True

    elif method == 'distance':
        MAX_DISTANCE = 3
        AVG_DISTANCE = 1.3

        def shortest_path_distance(start, end):
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                node, distance = queue.popleft()
                if node == end: return distance
                if distance >= MAX_DISTANCE: continue
                for edge, neighbor in adjacency_list[node] + adjacency_list_inv[node]:
                    edge = graph.graph.relation[edge].item()
                    if not node_rule_filtering(node, edge, neighbor): continue
                    if not edge_rule_filtering(node, edge, neighbor): continue
                    if neighbor not in visited and neighbor not in subgraph:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

            return float('inf')

        distances = []
        for node1 in subgraph:
            if node1 in other:
                distance = 0
            else:
                distance = []
                for node2 in other:
                    di = shortest_path_distance(node1, node2)
                    distance.append(di)
                distance = np.min(distance)
            distances.append(distance)

        print("Distances to existing subgraph:", distances)
        if np.mean(distances) <= AVG_DISTANCE:
            return True

    elif method == 'llm':
        text_subgraph = "\n".join(graph.id2entity[node] for node in conditions_in_subgraph)
        text_other = "\n".join(graph.id2entity[node] for node in conditions_in_other)

        prompt = f"""
Consider the following two set of conditions and reason whether they are connected clinically.

Condition Set A:
{text_subgraph}

Condition Set B:
{text_other}

Please provide the following reasoning steps:
1. Summarize the above two sets individually: [Your Summary]
2. Decide whether they are generally connected to the same disease or diagnoses: [Your Comparison]
3. Provide your answer: [YES or NO]
"""

        def validate_output(prob, response, output):
            assert ('YES' in response and "NO" not in response) or ('YES' not in response and "NO" in response)
            return prob, response, output

        if 'llm' in kwargs:
            llm = kwargs['llm']
        else:
            llm = LocalModel({"model": "D:\\models\\Qwen2.5-3B-Instruct"})
            llm.set_generation_config('do_sample', False)
            llm.set_generation_config('max_new_tokens', 1024)

        _, response, _, _ = llm.understand(prompt, post_process=validate_output)
        print(response.replace('\n', ''))
        return 'YES' in response

    return False


def get_connected_subgraphs(queries, included_nodes, sharable_nodes,
                            max_path_length, relevant_kwargs,
                            merge_method=None, verbose=False,
                            **kwargs):
    concept2idx = {}
    idx2subgraph = {}
    idx2name = {}

    for query in queries:
        # skip visited nodes
        if query in concept2idx: continue
        included_nodes = [n for n in included_nodes if n not in concept2idx]

        subgraph = get_subgraph(query, included_nodes, relevant_kwargs=relevant_kwargs, MAX_PATH_LENGTH=max_path_length)
        if not subgraph: continue

        # if a node is sharable, do NOT mark as visited
        # otherwise, mark as visited and update concept2idx
        visited_in_subgraph = [node for node in get_nodes(subgraph) if graph.id2entity[node] not in sharable_nodes]

        if verbose:
            print_subgraph(subgraph, starting_node=graph.id2entity[query])

        # whether to merge subgraphs
        max_connectivity_id = -1
        if merge_method is not None:  # search for the most connected existing subgraph
            max_connectivity = 0
            for subgraph_id, existing_subgraph in idx2subgraph.items():
                connectivity = get_subgraph_connectivity(subgraph, existing_subgraph, method=merge_method, **kwargs)
                if connectivity > max_connectivity:
                    max_connectivity = connectivity
                    max_connectivity_id = subgraph_id

        # if possible, merge new `subgraph` to existing subgraph
        if max_connectivity_id != -1:  # merge
            print(f"Merge into subgraph {max_connectivity_id}")
            max_connectivity_subgraph = idx2subgraph[max_connectivity_id]
            for new_node, new_edges in subgraph.items():
                if new_node in max_connectivity_subgraph:
                    max_connectivity_subgraph[new_node].extend(new_edges)
                else:
                    max_connectivity_subgraph[new_node] = new_edges
            for node in visited_in_subgraph:
                concept2idx[node] = max_connectivity_id

        else:  # add new subgraph
            idx = len(idx2subgraph)
            idx2subgraph[idx] = subgraph
            idx2name[idx] = query
            for node in visited_in_subgraph:
                concept2idx[node] = idx

    # phase output
    clusters = []
    for idx, subgraph in idx2subgraph.items():
        nodes = get_nodes(subgraph)

        if nodes:
            nodes = [graph.id2entity[c] for c in nodes]
            clusters.append({"header": graph.id2entity[idx2name[idx]],
                             "conditions": [c for c in nodes if is_C(c)],
                             "procedures": [c for c in nodes if is_P(c)],
                             "drugs": [c for c in nodes if is_D(c)],
                             "subgraph": subgraph})

    return clusters


def print_cluster(cluster):
    print(f"```subgraph: {cluster['header']}")
    if cluster["conditions"]:
        print("Conditions:", cluster["conditions"])
    if cluster["procedures"]:
        print("Procedures:", cluster["procedures"])
    if cluster["drugs"]:
        print("Medications:", cluster["drugs"])
    print("```")
