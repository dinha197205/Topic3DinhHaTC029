
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

# Step 1: Load the SPLADE model
model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)

# Step 2: Use the same documents as BM25 and Dense
documents = [
    "Wafer shortages may delay semiconductor production.",
    "The company approved a new share repurchase program.",
    "A shortage of silicon wafers can delay chip fabrication at foundries.",
    "The semiconductor company reported strong revenue growth."
]

# Step 3: Generate SPLADE embeddings
embeddings = list(model.embed(documents))

# Step 4: Initialize Qdrant Local
client = QdrantClient(path="./qdrant_splade_local")
collection_name = "splade_practice"

# Step 5: Create the sparse vector collection
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

# Step 6: Store document embeddings
points = [
    models.PointStruct(
        id=i + 1,
        vector={
            "sparse": models.SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist()
            )
        },
        payload={"text": document}
    )
    for i, (document, embedding)
    in enumerate(zip(documents, embeddings))
]

client.upsert(
    collection_name=collection_name,
    points=points
)

# Step 7: Verify stored data
print("COLLECTION:", collection_name)
print("NUMBER OF DOCUMENTS:", len(documents))
print("DOCUMENT 3 ACTIVE DIMENSIONS:", len(embeddings[2].indices))
print("STORED POINTS:", client.count(
    collection_name=collection_name,
    exact=True
).count)
