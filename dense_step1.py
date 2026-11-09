
from fastembed import TextEmbedding

# Buoc 1: Khoi tao mo hinh Dense
model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Buoc 2: Tao tai lieu mau
document = "Wafer shortages may delay semiconductor production."

# Buoc 3: Chuyen van ban thanh Dense Vector
embedding = next(model.embed([document]))

# Buoc 4: Kiem tra ket qua
print("DOCUMENT:", document)
print("VECTOR DIMENSION:", len(embedding))
print("FIRST 10 VALUES:", embedding[:10].tolist())

# Buoc 5: Tao vector cho Query
query = "What could disrupt semiconductor manufacturing?"

query_text = (
    "Represent this sentence for searching relevant passages: "
    + query
)

query_embedding = next(model.embed([query_text]))

print("QUERY:", query)
print("QUERY VECTOR DIMENSION:", len(query_embedding))
print("QUERY FIRST 10 VALUES:", query_embedding[:10].tolist())

# Buoc 6: Calculate cosine similarity
import numpy as np

dot_product = np.dot(query_embedding, embedding)

query_norm = np.linalg.norm(query_embedding)
document_norm = np.linalg.norm(embedding)

cosine_similarity = dot_product / (query_norm * document_norm)

print("COSINE SIMILARITY:", float(cosine_similarity))

# Step 7: Compare with an unrelated document

unrelated_document = (
    "The company approved a new share repurchase program."
)

unrelated_embedding = next(model.embed([unrelated_document]))

unrelated_similarity = np.dot(
    query_embedding, unrelated_embedding
) / (
    np.linalg.norm(query_embedding)
    * np.linalg.norm(unrelated_embedding)
)

print("\n--- SIMILARITY COMPARISON ---")
print("Query:", query)
print("Document A:", document)
print("Score A:", float(cosine_similarity))
print("Document B:", unrelated_document)
print("Score B:", float(unrelated_similarity))
