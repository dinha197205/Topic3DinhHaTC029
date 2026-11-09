
from qdrant_client import QdrantClient, models

# Buoc 1: Khoi tao Qdrant Local
client = QdrantClient(":memory:")

# Buoc 2: Tao collection cho BM25
client.create_collection(
    collection_name="bm25_practice",
    vectors_config={},
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)

# Buoc 3: Kiem tra collection
print(
    "COLLECTION EXISTS:",
    client.collection_exists("bm25_practice")
)

print(
    "POINT COUNT:",
    client.count(
        collection_name="bm25_practice",
        exact=True
    ).count
)

from fastembed import SparseTextEmbedding

# Buoc 4: Khoi tao FastEmbed BM25
model = SparseTextEmbedding(
    model_name="Qdrant/bm25"
)

# Buoc 5: Tao tai lieu va sparse vector
document = "NVIDIA H200 H200 revenue"

embedding = next(model.embed([document]))

# Buoc 6: Tao Point va luu vao Qdrant
point = models.PointStruct(
    id=1,
    vector={
        "bm25": models.SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist()
        )
    },
    payload={
        "text": document
    }
)

client.upsert(
    collection_name="bm25_practice",
    points=[point]
)

# Buoc 7: Kiem tra so Point sau khi luu
print(
    "POINT COUNT AFTER UPSERT:",
    client.count(
        collection_name="bm25_practice",
        exact=True
    ).count
)

# Buoc 8: Doc lai Point da luu trong Qdrant
stored_points = client.retrieve(
    collection_name="bm25_practice",
    ids=[1],
    with_payload=True,
    with_vectors=True
)

# Lay Point dau tien
stored_point = stored_points[0]

# In thong tin Point
print("STORED POINT ID:", stored_point.id)

print("STORED TEXT:", stored_point.payload["text"])

print("STORED VECTOR:", stored_point.vector["bm25"])

# Buoc I.2: Them 3 tai lieu moi

documents = [
    "NVIDIA H200 supply constraints may affect product availability.",
    "NVIDIA supply constraints reduced shipments.",
    "Apple reported revenue growth."
]

# Chuyen 3 tai lieu thanh sparse vectors
embeddings = list(model.embed(documents))

points = []

for i, (text, emb) in enumerate(
    zip(documents, embeddings),
    start=2
):
    point = models.PointStruct(
        id=i,
        vector={
            "bm25": models.SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist()
            )
        },
        payload={"text": text}
    )

    points.append(point)

# Luu 3 Point vao Qdrant
client.upsert(
    collection_name="bm25_practice",
    points=points
)

# Kiem tra so luong tai lieu
print(
    "TOTAL DOCUMENTS:",
    client.count(
        collection_name="bm25_practice",
        exact=True
    ).count
)

# Buoc 9: Tao cau truy van
query = "NVIDIA H200 supply constraints"

# Buoc 10: Chuyen query thanh sparse vector
query_embedding = next(model.query_embed(query))

print("QUERY:", query)
print("QUERY INDICES:", query_embedding.indices.tolist())
print("QUERY VALUES:", query_embedding.values.tolist())

# Buoc 11: Tim kiem BM25 trong Qdrant
response = client.query_points(
    collection_name="bm25_practice",
    query=models.SparseVector(
        indices=query_embedding.indices.tolist(),
        values=query_embedding.values.tolist()
    ),
    using="bm25",
    limit=3,
    with_payload=True
)

# Buoc 12: In ket qua tim kiem
for point in response.points:
    print("RESULT ID:", point.id)
    print("BM25 SCORE:", point.score)
    print("RESULT TEXT:", point.payload["text"])
