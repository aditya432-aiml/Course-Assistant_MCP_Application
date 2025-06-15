# retrieve_and_query.py

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ========== CONFIG ==========
VECTOR_DIR = "Data_Extraction_and_Retrival/vector_store"
FAISS_INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(VECTOR_DIR, "chunks.pkl")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ========== LOAD ==========
def load_vector_store():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve_chunks(question, model, index, chunks, top_k=5):
    query_embedding = model.encode([question], convert_to_tensor=False).astype("float32")
    D, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]

# ========== MAIN ==========
def main():
    print("🔍 Loading vector store and embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index, chunks = load_vector_store()

    while True:
        question = input("\n💬 Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        top_chunks = retrieve_chunks(question, model, index, chunks, top_k=5)
        print("\n📌 Top Relevant Chunks:\n")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"--- Chunk {i} ---\n{chunk}\n")

if __name__ == "__main__":
    main()