import subprocess
from sentence_transformers import SentenceTransformer
import chromadb

# -----------------------------
# 1️⃣ Initialize embedding model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 2️⃣ Sample documents
# -----------------------------
documents = [
    "RAG stands for Retrieval-Augmented Generation. It combines retrieval of documents with LLM generation.",
    "The capital of France is Paris.",
    "The GeForce GTX 1650 is a budget-friendly GPU by NVIDIA.",
    "My name is Arun",
    "today is sept 9th 2025",
]

# -----------------------------
# 3️⃣ Create embeddings
# -----------------------------
embeddings = embedder.encode(documents).tolist()

# -----------------------------
# 4️⃣ Setup ChromaDB
# -----------------------------
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_docs")
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("✅ Documents stored in ChromaDB")

# -----------------------------
# 5️⃣ Function to query ChromaDB
# -----------------------------
def retrieve_docs(query, top_k=2):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    # Flatten the list
    retrieved_docs = [d for sublist in results["documents"] for d in sublist]
    return retrieved_docs

# -----------------------------
# 6️⃣ Function to call Ollama Llama 3
# -----------------------------
def ask_llama(query, context_docs):
    # Combine retrieved documents + user query
    prompt = "Use the following documents to answer the question.\n\n"
    prompt += "\n".join(context_docs) + "\n\n"
    prompt += f"Question: {query}\nAnswer:"

    # Call Ollama via subprocess
    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# -----------------------------
# 7️⃣ Interactive chat loop
# -----------------------------
print("\n💬 Local RAG Chatbot (type 'exit' to quit)")
while True:
    user_query = input("\nYou: ")
    if user_query.lower() in ["exit", "quit"]:
        break
    docs = retrieve_docs(user_query)
    answer = ask_llama(user_query, docs)
    print("\nBot:", answer)
