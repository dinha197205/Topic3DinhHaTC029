
from fastembed import SparseTextEmbedding

# Step 1: Load the SPLADE model
model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)

# Step 2: Prepare an English document
document = (
    "A shortage of silicon wafers can delay "
    "chip fabrication at foundries."
)

# Step 3: Generate learned sparse embedding
embedding = next(model.embed([document]))

# Step 4: Display embedding information
print("DOCUMENT:", document)
print("VOCABULARY SIZE: 30522")
print("ACTIVE DIMENSIONS:", len(embedding.indices))

# Step 5: Display top-weighted token dimensions
pairs = sorted(
    zip(embedding.indices, embedding.values),
    key=lambda pair: float(pair[1]),
    reverse=True
)

print("\n--- TOP 10 WEIGHTED DIMENSIONS ---")

for index, value in pairs[:10]:
    print(f"INDEX: {int(index)} | WEIGHT: {float(value):.6f}")

from huggingface_hub import hf_hub_download

# Step 6: Download the model vocabulary
vocab_path = hf_hub_download(
    repo_id="prithivida/Splade_PP_en_v1",
    filename="vocab.txt"
)

# Step 7: Load tokens in vocabulary ID order
with open(vocab_path, "r", encoding="utf-8") as file:
    vocabulary = file.read().splitlines()

# Step 8: Decode the top-weighted token IDs
print("\n--- TOP 10 DECODED TOKENS ---")
print("ACTIVE DIMENSIONS:", len(embedding.indices))

for index, weight in pairs[:10]:
    token_id = int(index)
    token = vocabulary[token_id]

    print(
        f"TOKEN: {token:18} | "
        f"INDEX: {token_id:5} | "
        f"WEIGHT: {float(weight):.6f}"
    )
    
# Step 9: Inspect specific vocabulary terms

active_weights = {
    int(index): float(weight)
    for index, weight in zip(
        embedding.indices,
        embedding.values
    )
}

target_tokens = [
    "delay",
    "delayed",
    "semiconductor",
    "manufacturing",
    "production"
]

print("\n--- EXPANSION CHECK ---")

for token in target_tokens:
    if token in vocabulary:
        token_id = vocabulary.index(token)
        weight = active_weights.get(token_id, 0.0)

        print(
            f"TOKEN: {token:15} | "
            f"WEIGHT: {weight:.6f}"
        )
    else:
        print(f"TOKEN: {token:15} | NOT IN VOCABULARY")

# Step 10: Encode the query with SPLADE
query = "What could disrupt semiconductor manufacturing?"

query_embedding = next(model.embed([query]))

query_weights = {
    int(index): float(weight)
    for index, weight in zip(
        query_embedding.indices,
        query_embedding.values
    )
}

# Step 11: Find overlapping sparse dimensions
shared_indices = set(active_weights) & set(query_weights)

print("\n--- SPLADE QUERY-DOCUMENT OVERLAP ---")
print("QUERY:", query)
print("QUERY ACTIVE DIMENSIONS:", len(query_embedding.indices))
print("SHARED DIMENSIONS:", len(shared_indices))

# Step 12: Inspect shared terms and dot-product contributions
contributions = sorted(
    [
        (
            vocabulary[index],
            active_weights[index],
            query_weights[index],
            active_weights[index] * query_weights[index]
        )
        for index in shared_indices
    ],
    key=lambda item: item[3],
    reverse=True
)

for token, doc_weight, query_weight, contribution in contributions[:10]:
    print(
        f"TOKEN: {token:15} | "
        f"DOC: {doc_weight:.4f} | "
        f"QUERY: {query_weight:.4f} | "
        f"CONTRIBUTION: {contribution:.4f}"
    )

print(
    "SPARSE DOT PRODUCT:",
    sum(item[3] for item in contributions)
)
