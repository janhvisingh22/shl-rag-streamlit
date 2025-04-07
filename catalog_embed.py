import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Load the SHL catalog
df = pd.read_csv("shl_catalog.csv")

# 🧼 Clean column names (remove extra spaces, etc.)
df.columns = df.columns.str.strip()

# ✅ Confirm 'Text' exists
print("Available columns:", df.columns)

# Load model and encode the text column
model = SentenceTransformer('all-MiniLM-L6-v2')
catalog_embeddings = model.encode(df["Text"].tolist())

# Save embeddings and index
np.save("catalog_embeddings.npy", catalog_embeddings)

index = NearestNeighbors(n_neighbors=5, metric='cosine')
index.fit(catalog_embeddings)

with open("nn_index.pkl", "wb") as f:
    pickle.dump(index, f)

print("✅ Embeddings and index saved successfully!")
