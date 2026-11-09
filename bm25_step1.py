
from fastembed import SparseTextEmbedding

# Buoc 1: Khoi tao bo tao BM25 embedding
model = SparseTextEmbedding(
    model_name="Qdrant/bm25"
)

# Buoc 2: Tao mot tai lieu mau
document = "NVIDIA H200 H200 revenue"

# Buoc 3: Chuyen tai lieu thanh sparse vector
embedding = next(model.embed([document]))

# Buoc 4: In ket qua
print("DOCUMENT:", document)
print("INDICES:", embedding.indices.tolist())
print("VALUES:", embedding.values.tolist())

# Kiem tra ID cua tung tu rieng biet
words = ["NVIDIA", "H200", "revenue"]

for word, emb in zip(words, model.embed(words)):
    print(
        f"WORD: {word} | "
        f"INDICES: {emb.indices.tolist()}"
    )
