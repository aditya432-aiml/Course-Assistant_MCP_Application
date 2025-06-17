# retrival_with_docker_runner_llm.py

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ========== Client Setup ==========

client = OpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="docker"
)

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
        # print("\n📌 Top Relevant Chunks:\n")
        # for i, chunk in enumerate(top_chunks, 1):
            # print(f"--- Chunk {i} ---\n{chunk}\n")
            # Use only the most relevant chunk
        if not top_chunks:
            print("⚠️ No relevant chunks found.")
            continue
        combined_chunks = "\n\n".join(top_chunks)
        print(f"--- Combined Top 5 Chunks ---\n{combined_chunks}\n")
        # Optionally, you can use the OpenAI client to process the question further
        response = client.chat.completions.create(
            model="ai/llama3.2:1B-Q4_0",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a expert at explaining the following content: {combined_chunks}. If grretings are present, ignore them."
                },
                {
                    "role": "user", 
                    "content": question
                }
            ],
            stream = True
        )
        # print(f"🤖 Response: {response.choices[0].message.content}")
        # for choice1 in response:
        #     print(f"🤖 Response: {choice1.choices[0].delta.content}",end="")
        for choice1 in response:
            content = getattr(choice1.choices[0].delta, "content", "")
            if content:
                print(content, end="", flush=True)
        print()  # Print newline after streaming response

if __name__ == "__main__":
    main()