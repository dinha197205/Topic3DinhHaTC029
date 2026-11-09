
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# Step 1: Load the same Dense model
model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Step 2: Connect to the existing Qdrant database
client = QdrantClient(path="./qdrant_dense_local")
collection_name = "dense_practice"

# Step 3: Prepare the search query
query = "What could disrupt semiconductor manufacturing?"

query_text = (
    "Represent this sentence for searching relevant passages: "
    + query
)

# Step 4: Generate a 384-dimensional query embedding
query_embedding = next(model.embed([query_text]))

# Step 5: Search the existing collection
response = client.query_points(
    collection_name=collection_name,
    query=query_embedding.tolist(),
    using="dense",
    limit=4,
    with_payload=True
)

# Step 6: Display ranked results
print("QUERY:", query)
print("QUERY VECTOR DIMENSION:", len(query_embedding))
print("\n--- DENSE TOP-4 RESULTS ---")

for rank, point in enumerate(response.points, start=1):
    print(f"\nRANK: {rank}")
    print(f"POINT ID: {point.id}")
    print(f"COSINE SCORE: {point.score:.6f}")
    print(f"DOCUMENT: {point.payload['text']}")
