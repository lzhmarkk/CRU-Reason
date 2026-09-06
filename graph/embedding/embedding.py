import json
import pickle
import tiktoken
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

from graph.graphloader import read_graph_from_file
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the API key
api_key_file = Path('./utils/apis/anywhere.key')
api_key = json.loads(api_key_file.read_text())
client = OpenAI(api_key=api_key["key"], base_url=api_key["url"])


def generate_embeddings(texts, model="text-embedding-3-large", dimension=None, num_threads=1):
    embeddings = {}

    def process_text(text):
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        if len(tokens) > 8192:
            text = encoding.decode(tokens[-8192:])

        if dimension:
            embedding = client.embeddings.create(input=text,
                                                 model=model,
                                                 dimensions=dimension)
        else:
            embedding = client.embeddings.create(input=text,
                                                 model=model)
        embedding = embedding.data[0].embedding
        return text, embedding

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_text, text) for text in texts]
            for future in tqdm(as_completed(futures), total=len(texts)):
                text, embedding = future.result()
                embeddings[text] = embedding
    else:
        for text in texts:
            text, embedding = process_text(text)
            embeddings[text] = embedding

    return embeddings


if __name__ == '__main__':
    num_threads = 10
    EMB_DIMENSIONS = 1024

    print("Reading knowledge graph data...")
    edges = read_graph_from_file("./graph/kg_raw.txt")
    entities = list(set([e[0] for e in edges] + [e[2] for e in edges]))
    relations = list(set(e[1] for e in edges))

    # entity embedding
    entity_embeddings_file = f"./graph/embedding/entity_embeddings.pkl"
    print(f"Generating embeddings for {len(entities)} entities using {num_threads} threads...")
    entity_embeddings = generate_embeddings(entities, dimension=EMB_DIMENSIONS, num_threads=num_threads)
    with open(entity_embeddings_file, 'wb') as f:
        pickle.dump(entity_embeddings, f)

    # relation embedding
    relation_embeddings_file = f"./graph/embedding/relation_embeddings.pkl"
    print(f"Generating embeddings for {len(relations)} relations using {num_threads} threads...")
    relation_embeddings = generate_embeddings(relations, dimension=EMB_DIMENSIONS, num_threads=num_threads)
    with open(relation_embeddings_file, 'wb') as f:
        pickle.dump(relation_embeddings, f)
