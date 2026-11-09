
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

# Step 1: Use the same documents as Dense
documents = [
    "Wafer shortages may delay semiconductor production.",
    "The company approved a new share repurchase program.",
    "A shortage of silicon wafers can delay chip fabrication at foundries.",
    "The semiconductor company reported strong revenue growth."
]

query = "What could disrupt semiconductor manufacturing?"

# Step 2: Initialize BM25
model = SparseTextEmbedding(model_name="Qdrant/bm25")

# Step 3: Create a temporary Qdrant database
client = QdrantClient(":memory:")
collection_name = "bm25_comparison"

client.create_collection(
    collection_name=collection_name,
    vectors_config={},
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)

# Step 4: Convert documents into sparse vectors
embeddings = list(model.embed(documents))

points = [
    models.PointStruct(
        id=i + 1,
        vector={
            "bm25": models.SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist()
            )
        },
        payload={"text": document}
    )
    for i, (document, embedding)
    in enumerate(zip(documents, embeddings))
]

client.upsert(collection_name=collection_name, points=points)

# Step 5: Encode query and search
query_vector = next(model.query_embed(query))

response = client.query_points(
    collection_name=collection_name,
    query=models.SparseVector(
        indices=query_vector.indices.tolist(),
        values=query_vector.values.tolist()
    ),
    using="bm25",
    limit=4,
    with_payload=True
)

# Step 6: Display results
print("QUERY:", query)
print("\n--- BM25 RESULTS ---")

for rank, point in enumerate(response.points, start=1):
    print(f"\nRANK: {rank}")
    print(f"POINT ID: {point.id}")
    print(f"BM25 SCORE: {point.score:.6f}")
    print(f"DOCUMENT: {point.payload['text']}")
