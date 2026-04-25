# rag_pipeline.py
# Este archivo contiene la lógica principal del sistema RAG:
# - Cargar el índice FAISS y los chunks
# - Buscar fragmentos relevantes para una pregunta
# - Construir el prompt con el contexto recuperado
# - Generar la respuesta usando Google Gemini

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()

# Archivos generados por ingest.py
CHUNKS_FILE     = "chunks.json"
INDEX_FILE      = "index.faiss"

# Modelo de embeddings (debe ser el mismo que se usó en ingest.py)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Número de fragmentos a recuperar
TOP_K = 5


def load_resources():
    """Carga el índice FAISS, los chunks y el modelo de embeddings."""
    # Cargar chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Cargar índice FAISS
    index = faiss.read_index(INDEX_FILE)

    # Cargar modelo de embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)

    return chunks, index, model


def retrieve(query: str, chunks: list, index, model, top_k: int = TOP_K) -> list[dict]:
    """Busca los fragmentos más relevantes para la pregunta."""

    # Convertir la pregunta a embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Buscar en el índice FAISS
    distances, indices = index.search(query_embedding, top_k)

    # Recuperar los chunks correspondientes
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["score"] = float(distances[0][i])
            results.append(chunk)

    return results


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Construye el prompt con el contexto recuperado."""

    context = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += f"\n--- Fragmento {i} (Fuente: {chunk['source']}) ---\n"
        context += chunk["text"]
        context += "\n"

    prompt = f"""Eres un asistente técnico especializado en códigos de error de BMW.
Responde únicamente con base en el contexto recuperado que se te proporciona a continuación.

Si no hay evidencia suficiente en los fragmentos recuperados, responde exactamente:
"No tengo evidencia suficiente en los documentos recuperados para responder esta pregunta."

Siempre cita las fuentes que utilizaste al final de tu respuesta.

CONTEXTO RECUPERADO:
{context}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""

    return prompt


def generate_response(prompt: str) -> str:
    """Genera la respuesta usando Groq (llama3-8b-8192)."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ Error: No se encontró la API key de Groq en el archivo .env"

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )

    return response.choices[0].message.content


def ask(query: str, chunks: list, index, model, top_k: int = TOP_K) -> dict:
    """
    Función principal del RAG.
    Recibe una pregunta y devuelve la respuesta con sus fuentes.
    """

    # 1. Recuperar fragmentos relevantes
    retrieved = retrieve(query, chunks, index, model, top_k)

    # 2. Construir el prompt
    prompt = build_prompt(query, retrieved)

    # 3. Generar respuesta con Gemini
    answer = generate_response(prompt)

    # 4. Preparar resultado estructurado
    result = {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "chunk_id": c["chunk_id"],
                "source":   c["source"],
                "fragment": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                "score":    round(c["score"], 4)
            }
            for c in retrieved
        ]
    }

    return result
