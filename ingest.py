# ingest.py

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Carpeta donde están los archivos TXT
DATA_FOLDER = "data"

# Archivos donde se guardarán los chunks y el índice
CHUNKS_FILE = "chunks.json"
INDEX_FILE  = "index.faiss"

# Modelo de embeddings (se descarga automáticamente la primera vez)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Configuración de chunking ──────────────────────────────────────────────────
CHUNK_SIZE    = 300   # Tamaño del fragmento en caracteres
CHUNK_OVERLAP = 50    # Solapamiento entre fragmentos


def load_documents(folder: str) -> list[dict]:
    """Carga todos los archivos TXT de la carpeta data."""
    documents = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "filename": filename,
                "content": content
            })
            print(f" Cargado: {filename}")
    return documents


def split_into_chunks(documents: list[dict]) -> list[dict]:
    """Divide cada documento en fragmentos con solapamiento."""
    chunks = []
    for doc in documents:
        text    = doc["content"]
        source  = doc["filename"]
        start   = 0
        idx     = 0

        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end]

            # Solo guardar chunks con contenido real
            if chunk.strip():
                chunks.append({
                    "chunk_id": f"{source}_chunk_{idx}",
                    "source":   source,
                    "text":     chunk
                })
                idx += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def generate_embeddings(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Genera embeddings para cada chunk."""
    texts = [c["text"] for c in chunks]
    print(f"\n  Generando embeddings para {len(texts)} fragmentos...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Crea un índice FAISS con los embeddings."""
    dimension = embeddings.shape[1]
    index     = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_chunks(chunks: list[dict], filepath: str):
    """Guarda los chunks en un archivo JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def run_ingestion():
    """Ejecuta el pipeline completo de ingesta."""
    print("=" * 50)
    print("  PIPELINE DE INGESTA - BMW RAG")
    print("=" * 50)

    # 1. Cargar documentos
    print("\n Cargando documentos...")
    documents = load_documents(DATA_FOLDER)
    print(f"  Total documentos cargados: {len(documents)}")

    # 2. Dividir en chunks
    print(f"\n Dividiendo en fragmentos (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_into_chunks(documents)
    print(f"  Total fragmentos generados: {len(chunks)}")

    # 3. Cargar modelo de embeddings
    print(f"\n Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 4. Generar embeddings
    embeddings = generate_embeddings(chunks, model)
    print(f"  Dimensión de embeddings: {embeddings.shape[1]}")

    # 5. Construir índice FAISS
    print("\n  Construyendo índice FAISS...")
    index = build_faiss_index(embeddings)
    print(f"  Vectores indexados: {index.ntotal}")

    # 6. Guardar chunks e índice
    print("\n Guardando chunks e índice...")
    save_chunks(chunks, CHUNKS_FILE)
    faiss.write_index(index, INDEX_FILE)
    print(f"   Chunks guardados en: {CHUNKS_FILE}")
    print(f"   Índice guardado en:  {INDEX_FILE}")

    print("\n Ingesta completada exitosamente.")
    print("=" * 50)


if __name__ == "__main__":
    run_ingestion()
