import pickle
import numpy as np

with open('Sources/Embeddings/user_personalized_preferences.pkl', "rb") as f:
  embeddings = pickle.load(f)

first_key = next(iter(embeddings))
first_embedding = embeddings[first_key]

user_id = next(iter(embeddings))
user_embedding = embeddings[user_id]

print("User:", user_id)
print("User embedding keys:", user_embedding.keys())

first_subkey = next(iter(user_embedding))
print("Sub-embedding key:", first_subkey)
print("Sub-embedding type:", type(user_embedding[first_subkey]))
print("Sub-embedding shape:", np.array(user_embedding[first_subkey]).shape)
