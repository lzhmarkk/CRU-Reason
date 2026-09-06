import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.cross_encoder import CrossEncoder


class UMLSBERT:
    def __init__(self, model_name_or_path):
        print("Loading UMLSBERT model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, device_map='auto', trust_remote_code=True)
        self.device = self.model.device
        print("UMLSBERT model and tokenizer loaded successfully.")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def batch_encode(self, texts, batch_size=16):
        print(f"Starting batch encoding of {len(texts)} texts with batch size {batch_size}.")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                model_output = self.model(**inputs)  # Slow
                attention_mask = inputs["attention_mask"]
                batch_embeddings = self.mean_pooling(model_output, attention_mask)
            all_embeddings.extend(batch_embeddings.cpu().numpy())
        return np.array(all_embeddings)


class UMLS_CrossEncoder:
    def __init__(self, model_name_or_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(model_name_or_path, max_length=512, device=device)
        print("UMLS CrossEncoder loaded successfully.")

    def score(self, query, rels):
        if not rels:
            return []
        pairs = [[query, rel] for rel in rels]
        scores = self.model.predict(pairs).tolist()
        return scores
