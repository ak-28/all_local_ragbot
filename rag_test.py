from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load embedding model (lightweight, works on GTX 1650)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Sample documents
documents = [
    "RAG stands for Retrieval-Augmented Generation.",
    "The capital of France is Paris.",
    "The GeForce GTX 1650 is a budget-friendly GPU by NVIDIA.",
    "My name is Arun and i like to be called V"
]

# 3. Create embeddings
embeddings = embedder.encode(documents).tolist()

# 4. Create ChromaDB client + collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_docs")

# 5. Insert documents
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("✅ Documents added to ChromaDB!")

# 6. Test a query
query = "What Gpu is budget friendly?"
query_embedding = embedder.encode([query]).tolist()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print("\n🔍 Query:", query)
print("Top match:", results["documents"][0][0])
