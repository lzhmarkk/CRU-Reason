import torch
from modules.kg import graph, adjacency_list, adjacency_list_inv
from modules.motif import is_C, is_D, is_P, get_type
from modules.blacklists import blacklist_entity
from collections import deque, defaultdict
from TherapyReason.utils.relevance import is_relevant_edge


def print_subgraph(subgraph, starting_node=""):
    print(f"```subgraph: {starting_node}")
    for node, neighbor in subgraph.items():
        for edge, dst in neighbor:
            src = graph.id2entity[node]
            dst = graph.id2entity[dst]
            print(f"{src} [{get_type(src)}]", "->", graph.id2relation[edge], "->", f"{dst} [{get_type(dst)}]")

    nodes = [graph.id2entity[node] for node in get_nodes(subgraph)]
    print("conditions", [node for node in nodes if is_C(node)])
    print("procedures", [node for node in nodes if is_P(node)])
    print("drugs", [node for node in nodes if is_D(node)])
    print("```")


def get_nodes(subgraph):
    nodes = set()
    for node, neighbor in subgraph.items():
        nodes.add(node)
        nodes.update(dst for _, dst in neighbor)
    return nodes


def get_subgraph(query, included_nodes, relevant_kwargs, MAX_PATH_LENGTH=2, order=False):
    assert query in included_nodes
    if  any(w in graph.id2entity[query].lower() for w in blacklist_entity):
        return None
    # print(f"Starting from node '{graph.id2entity[query]}'")

    # Initialize the subgraph as an adjacency list
    subgraph = defaultdict(list)
    subgraph[query] = []
    visited = {query}

    # Queue for BFS: stores tuples of (current_node, current_path_length)
    queue = deque([(query, 0)])

    # Perform BFS
    while queue:
        current, path_length = queue.popleft()

        # Explore all neighbors
        # print("current", current)
        # candidates = set(adjacency_list[current] + adjacency_list_inv[current])
        for direction, candidates in {'forward': adjacency_list[current], 'backward': adjacency_list_inv[current]}.items():
            # cond first, then proc, drug last
            if order:
                candidate_cond = [x for x in candidates if is_C(graph.id2entity[x[1]])]
                candidate_proc = [x for x in candidates if is_P(graph.id2entity[x[1]])]
                candidate_drug = [x for x in candidates if is_D(graph.id2entity[x[1]])]
                candidates = candidate_cond + candidate_proc + candidate_drug

            for edge, next in candidates:
                edge = graph.graph.relation[edge].item()

                # visited
                if next in visited: continue
                if next not in included_nodes: continue

                # If the current path length exceeds the maximum length
                if path_length >= MAX_PATH_LENGTH:  continue

                # If last node is condition
                # if path_length + 1 >= MAX_PATH_LENGTH and is_C(graph.id2entity[next]): continue

                # If (current, edge, next) is not valid
                if not is_relevant_edge(current, edge, next,
                                        existing_nodes=visited,
                                        direction=direction,
                                        **relevant_kwargs):
                    continue

                # Add the valid edge to the subgraph
                subgraph[current].append((edge, next))
                # print(f"Adding '{graph.id2entity[next]}' to subgraph {[graph.id2entity[n] for n in nodes_in_subgraph]}")
                visited.add(next)

                # Add the neighbor (next) to the queue with incremented path length
                queue.append((next, path_length + 1))

    return dict(subgraph)


def get_subgraph_dfs(query, included_nodes, relevant_kwargs, MAX_PATH_LENGTH=2, order=False):
    assert query in included_nodes
    # print(f"Starting from node '{graph.id2entity[query]}'")

    # Initialize the subgraph as an adjacency list
    subgraph = defaultdict(list)
    subgraph[query] = []
    visited = set()

    def dfs_inner_function(current, path_length):
        if path_length >= MAX_PATH_LENGTH or current in visited:
            return

        visited.add(current)

        # Add the valid edge to the subgraph
        candidates = adjacency_list[current]

        # cond first, then proc, drug last
        if order:
            candidate_cond = [x for x in candidates if is_C(graph.id2entity[x[1]])]
            candidate_proc = [x for x in candidates if is_P(graph.id2entity[x[1]])]
            candidate_drug = [x for x in candidates if is_D(graph.id2entity[x[1]])]
            candidates = candidate_cond + candidate_proc + candidate_drug

        for edge, next in candidates:
            edge = graph.graph.relation[edge].item()

            # visited
            if next in visited: continue
            if next not in included_nodes: continue

            # If the current path length exceeds the maximum length
            if path_length >= MAX_PATH_LENGTH:  continue

            # If last node is condition
            # if path_length + 1 >= MAX_PATH_LENGTH and is_C(graph.id2entity[next]): continue

            # If (current, edge, next) is not valid
            if not is_relevant_edge(current, edge, next,
                                    existing_nodes=visited,
                                    direction='forward',
                                    **relevant_kwargs):
                continue

            subgraph[current].append((edge, next))

            if path_length < MAX_PATH_LENGTH:
                dfs_inner_function(next, path_length + 1)

    dfs_inner_function(query, 0)

    return dict(subgraph)


def get_subgraph_MLP(query, included_nodes, relevant_kwargs, MAX_PATH_LENGTH=2):
    assert query in included_nodes and is_C(graph.id2entity[query])
    print(f"Starting from node '{graph.id2entity[query]}'")

    # Initialize the subgraph as an adjacency list
    subgraph = defaultdict(list)
    subgraph[query] = []

    return subgraph
