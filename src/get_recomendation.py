from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from get_data import get_data


def load_embeddings_from_data():
    embeddings = np.load("data/embeddings_tunned.npy")
    return embeddings



def get_recommendation(query, threshold=0.1):
    df = get_data()
    embedder = SentenceTransformer("models/tunned_sbert_corona")
    query_embedding = embedder.encode(query)

    embeddings = load_embeddings_from_data()

    similarity_scores = embedder.similarity(query_embedding, embeddings)[0]

    scores, indices = torch.topk(similarity_scores, k=10)

    results = []
    for score, idx in zip(scores, indices):
        if score > threshold:
            print(df.iloc[int(idx)]["OriginalTweet"], "(Score: %.4f)" % (score))
            results.append(
                {
                    "OriginalTweet": df.iloc[int(idx)]["OriginalTweet"],
                    "Relevance": score.item(),
                }
            )
    return results


if __name__ == "__main__":
    query = "covid is a hoax"
    get_recommendation(query)
