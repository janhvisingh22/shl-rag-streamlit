import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Load the dataset and model
df = pd.read_csv("shl_catalog.csv")
df.columns = df.columns.str.strip()  # Remove whitespace
model = SentenceTransformer("all-MiniLM-L6-v2")
catalog_embeddings = np.load("catalog_embeddings.npy")

# Fit Nearest Neighbors index
index = NearestNeighbors(metric="cosine")
index.fit(catalog_embeddings)

def get_top_k(query, k=5):
    query_embedding = model.encode([query])  # Make it 2D
    
    # Get actual number of samples in index
    n_samples_fit = index._fit_X.shape[0]

    # Set k to the smaller of desired k and actual number of samples
    actual_k = min(k, n_samples_fit)

    distances, indices = index.kneighbors(query_embedding, n_neighbors=actual_k, return_distance=True)
    return df.iloc[indices[0]]

print("DF columns:", df.columns.tolist())

def generate_response(top_df):
    top_df.columns = top_df.columns.str.strip()
    # print actual column names for debugging
    print("DEBUG: top_df.columns =", top_df.columns)
    
    response = "Here are the top recommended assessments:\n\n"
    for _, row in top_df.iterrows():
        name = row.get('title', 'N/A')
        desc = row.get('description', 'No description available')
        response += f"- {name}: {desc}\n"
    return response
