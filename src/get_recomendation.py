import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.get_data import get_data


def load_embeddings_from_data():
    embeddings = np.load("data/embeddings_tunned.npy")
    return embeddings



def get_recommendation(query, threshold=0.25):
    df = get_data()
    embedder = SentenceTransformer("models/tunned_sbert_corona")
    query_embedding = embedder.encode(query)

    embeddings = load_embeddings_from_data()

    similarity_scores = embedder.similarity(query_embedding, embeddings)[0]

    scores, indices = torch.topk(similarity_scores, k=10)

    results = []
    for score, idx in zip(scores, indices):
        if score > threshold:
            # print(df.iloc[int(idx)]["OriginalTweet"], "(Score: %.4f)" % (score))
            results.append(
                {
                    "Document": int(idx),
                    "OriginalTweet": df.iloc[int(idx)]["OriginalTweet"],
                    "Relevance": score.item(),
                }
            )
    df_results = pd.DataFrame(results)
    return df_results
