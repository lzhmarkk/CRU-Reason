import os
import json
import itertools
import random
from graspologic.partition import hierarchical_leiden
from graph.graphloader import Graph
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from utils.llm import LLM
from graph.embedding.embedding import generate_embeddings

MAX_TRIPLES_PER_SUMMARY = 20
MAX_SUMMARY_PER_REQUEST = 3


def generate_community_summary(model: LLM, triples: list) -> str:
    # Format the triples as a bullet list
    format_triple = lambda triple: f"({triple[0]}, {triple[1]}, {triple[2]})"
    formatted_triples = "\n".join([f"- {format_triple(triple)}" for triple in triples])

    prompt = f"""\
You are a knowledgeable medical assistant tasked with generating a comprehensive summary of the medical concepts and relationships provided below.
Given a list of medical triples in the format (entity1, relationship, entity2), please create a single, coherent summary that captures the key medical knowledge represented by these concepts and their relationships. The summary should be written in the third person and include all the entity names for full context.
Focus on how these medical concepts are interconnected and their relevance to medical understanding or patient care. If there are any contradictions in the triples, please resolve them in the summary.
Please provide only the summary without any starting words or phrases, and without mentioning the individual triples.
The summary should be an integrated representation of the medical knowledge contained in the triples.

Example:
Triples:
- (Diabetes, is a risk factor for, Cardiovascular Disease)
- (Hypertension, is associated with, Diabetes)
- (Obesity, contributes to, Diabetes)

Summary:
Diabetes, hypertension, and obesity are closely interconnected medical conditions. Diabetes is a significant risk factor for developing cardiovascular disease. Hypertension, or high blood pressure, is often associated with diabetes. Obesity contributes to the development of diabetes, as excess body weight can lead to insulin resistance. Managing these conditions together is crucial for reducing the risk of serious complications and improving overall patient care.

Triples:
{formatted_triples}

Summary:
"""

    _, response, _ = model.understand(prompt)
    return response


def combine_summaries(model: LLM, summaries: list) -> str:
    prompt = f"""\
You are a knowledgeable medical assistant tasked with generating a comprehensive summary of the medical concepts and relationships based on the individual summaries provided below.
Given the following summaries of a community, please create a single, coherent summary that captures the key medical knowledge represented by this community. The combined summary should include all the important entities and relationships mentioned in the individual summaries.
Focus on the main concepts, their relationships, and their relevance to medical understanding or patient care. The summary should be concise yet informative, providing a comprehensive overview of the community's medical knowledge.
Please ensure that the combined summary is written in the third person and includes all the relevant entity names for full context.
Please provide only the summary without any starting words or phrases.

Summaries:
{summaries}

Combined Summary:
"""

    _, response, _ = model.understand(prompt)
    return response


def process_community(model, triples, i, level, run, randomness):
    if len(triples) > 150:
        seed_summaries = []
        intermediate_summaries = {}
        summary = "The community is too large"
    else:
        if len(triples) <= MAX_TRIPLES_PER_SUMMARY:
            summary = generate_community_summary(model, triples)
            seed_summaries = [summary]
            intermediate_summaries = {}
            print(f"Run: {run}, Level: {level}, Community ID: {i}")
        else:
            seed_summaries = [
                generate_community_summary(model, triples[j:j + MAX_TRIPLES_PER_SUMMARY])
                for j in range(0, len(triples), MAX_TRIPLES_PER_SUMMARY)
            ]
            for k, summary in enumerate(seed_summaries):
                print(f"Run: {run}, Level: {level}, Community ID: {i}, Part: {k}")

            intermediate_summaries = {}
            intermediate_level = 1

            summaries = seed_summaries
            while len(summaries) > MAX_SUMMARY_PER_REQUEST:
                new_summaries = []
                for j in range(0, len(summaries), MAX_SUMMARY_PER_REQUEST):
                    chunk_summaries = summaries[j:j + MAX_SUMMARY_PER_REQUEST]
                    new_summary = combine_summaries(model, chunk_summaries)
                    new_summaries.append(new_summary)

                intermediate_summaries[f"level {intermediate_level}"] = new_summaries
                print(f"Run: {run}, Level: {level}, Community ID: {i}, Intermediate Level: {intermediate_level}")

                summaries = new_summaries
                intermediate_level += 1

            summary = combine_summaries(model, summaries)
            print(f"Run: {run}, Level: {level}, Community ID: {i}, Combine")

    community_data = {
        "run": run,
        "level": level,
        "community_id": i,
        "triples": triples,
        "summaries": seed_summaries,
        "intermediate_summaries": intermediate_summaries,
        "summary": summary,
        "randomness": randomness,
        "summary_embedding": list(generate_embeddings([summary]).values())[0],
        "nodes": list(set(itertools.chain(*[[t[0], t[2]] for t in triples])))
    }

    return community_data


def partition_and_summarize(g, model, max_cluster_size, output_dir):
    N_UNIQUE_COMMUNITIES = 60000

    print(f"Partitioning graph with max_cluster_size: {max_cluster_size}")

    # Note: g.node_id = g.node['entity']
    networkx = to_networkx(g.graph, node_attrs=['entity'], edge_attrs=['relation'], to_undirected=True)

    unique_communities = set()
    run = 0
    output_data = []

    while len(unique_communities) < N_UNIQUE_COMMUNITIES:
        print(f"Running iteration {run}...")
        print(f"Unique communities: {len(unique_communities)}")
        random.seed(run)
        randomness = random.uniform(0.01, 1.0)
        community_mapping = hierarchical_leiden(networkx, max_cluster_size=max_cluster_size, random_seed=run + 1, randomness=randomness)

        node_to_community = {}
        for partition in community_mapping:
            level = partition.level
            node = partition.node
            community = partition.cluster

            if level not in node_to_community:
                node_to_community[level] = {}
            node_to_community[level][node] = community

        for level, node_to_community_map in node_to_community.items():
            print(f"Processing level {level} with {len(set(node_to_community_map.values()))} communities")

            unprocessed_communities = []

            for community_id in tqdm(set(node_to_community_map.values())):
                community_nodes = [n for n in networkx.nodes if node_to_community_map.get(n) == community_id]
                triples = [
                    (g.id2entity[u], g.id2relation[feat['relation']], g.id2entity[v])  # textual entity & relation name
                    for u, v, feat in networkx.edges(community_nodes, data=True)
                    if u in community_nodes and v in community_nodes
                ]

                triples_set = frozenset(triples)
                if triples_set not in unique_communities:
                    unique_communities.add(triples_set)
                    unprocessed_communities.append((list(triples), community_id, level, run, randomness))

                    if len(unique_communities) >= N_UNIQUE_COMMUNITIES:
                        break

            # Process unprocessed communities using multi-threading
            for item in unprocessed_communities:
                output_path = os.path.join(output_dir, f"{item[3]}-{item[1]}.json")  # {run}-{community_id}.json
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        community = json.load(f)
                else:
                    community = process_community(model, *item)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(community, f)

                output_data.append(community)

            if len(unique_communities) >= N_UNIQUE_COMMUNITIES:
                break

        run += 1
        break  # conduct only once to save costs

    print("Summaries extracted and saved successfully.")
    return output_data


def partition_and_summarize_cached(g: Graph, model: LLM, output_dir: str):
    MAX_COMMUNITY_SIZE = 10

    output_file = os.path.join(output_dir, "communities.json")
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            communities = json.load(f)

    else:
        output_dir = os.path.join(output_dir, "community")
        os.makedirs(output_dir, exist_ok=True)
        communities = partition_and_summarize(g, model, max_cluster_size=MAX_COMMUNITY_SIZE, output_dir=output_dir)

        # Save updated communities to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(communities, f)

    summary2nodes, summary2emb = {}, {}
    for community in communities:
        summary2nodes[community['summary']] = community['nodes']
        summary2emb[community['summary']] = community['summary_embedding']

    return communities, summary2nodes, summary2emb
