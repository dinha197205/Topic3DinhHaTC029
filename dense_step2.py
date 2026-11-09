
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

# Step 1: Initialize the embedding model
model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Step 2: Prepare sample documents
documents = [
    "Wafer shortages may delay semiconductor production.",
    "The company approved a new share repurchase program.",
    "A shortage of silicon wafers can delay chip fabrication at foundries.",
    "The semiconductor company reported strong revenue growth."
]

# Step 3: Generate dense embeddings
embeddings = list(model.embed(documents))

# Step 4: Initialize local Qdrant
client = QdrantClient(path="./qdrant_dense_local")
collection_name = "dense_practice"

# Step 5: Create a collection with cosine similarity
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        }
    )

# Step 6: Store document vectors
points = [
    models.PointStruct(
        id=i + 1,
        vector={"dense": embedding.tolist()},
        payload={"text": document}
    )
    for i, (document, embedding)
    in enumerate(zip(documents, embeddings))
]

client.upsert(
    collection_name=collection_name,
    points=points
)

# Step 7: Verify stored documents
print("COLLECTION:", collection_name)
print("NUMBER OF DOCUMENTS:", len(documents))
print("VECTOR DIMENSION:", len(embeddings[0]))
print("STORED POINTS:", client.count(
    collection_name=collection_name,
    exact=True
).count)
