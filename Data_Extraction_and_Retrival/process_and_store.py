# process_and_store.py

import os
from pathlib import Path
import re
import pickle
import numpy as np
import faiss

from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ========== CONFIG ==========
PDF_FOLDER = "Data_Training/15_06_2025/"
VECTOR_DIR = "Data_Extraction_and_Retrival/vector_store"
FAISS_INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(VECTOR_DIR, "chunks.pkl")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ========== UTILS ==========
def extract_text_from_pdf(pdf_path):
    elements = partition_pdf(str(pdf_path),strategy="auto")
    return "\n".join([el.text for el in elements if hasattr(el, "text")])

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def embed_chunks(chunks, model):
    return model.encode(chunks, convert_to_tensor=False)

def save_vector_store(index, chunks):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

# ========== MAIN ==========
def main():
    print("🔄 Starting PDF processing and vector storage...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    all_chunks, all_embeddings = [], []

    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDF files found in {PDF_FOLDER}")
        return

    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}")
        try:
            text = extract_text_from_pdf(pdf_file)
            if not text.strip():
                print(f"⚠️ Skipping {pdf_file.name}: No extractable text found.")
                continue

            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)
            if not chunks:
                print(f"⚠️ Skipping {pdf_file.name}: No chunks generated.")
                continue

            embeddings = embed_chunks(chunks, model)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")

    if not all_embeddings:
        print("⚠️ No valid embeddings found. Exiting without creating index.")
        return

    embedding_matrix = np.vstack(all_embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    save_vector_store(index, all_chunks)

    print("✅ FAISS index and chunks saved successfully!")

if __name__ == "__main__":
    main()