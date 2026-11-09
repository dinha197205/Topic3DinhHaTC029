
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

# Step 1: Load the SPLADE model
model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)

# Step 2: Connect to the existing Qdrant database
client = QdrantClient(path="./qdrant_splade_local")
collection_name = "splade_practice"

# Step 3: Prepare the search query
query = "What could disrupt semiconductor manufacturing?"

# Step 4: Generate the query sparse vector
query_embedding = next(model.embed([query]))

# Step 5: Search using SPLADE
response = client.query_points(
    collection_name=collection_name,
    query=models.SparseVector(
        indices=query_embedding.indices.tolist(),
        values=query_embedding.values.tolist()
    ),
    using="sparse",
    limit=4,
    with_payload=True
)

# Step 6: Display ranked results
print("QUERY:", query)
print("QUERY ACTIVE DIMENSIONS:", len(query_embedding.indices))
print("\n--- SPLADE TOP-4 RESULTS ---")

for rank, point in enumerate(response.points, start=1):
    print(f"\nRANK: {rank}")
    print(f"POINT ID: {point.id}")
    print(f"SPLADE SCORE: {point.score:.6f}")
    print(f"DOCUMENT: {point.payload['text']}")
