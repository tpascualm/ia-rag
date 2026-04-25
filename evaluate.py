# evaluate.py
# Evalúa el sistema RAG usando el golden_set.json
# Calcula Precision@k, Recall@k y genera un reporte con los resultados.

import json
import time
import pandas as pd
from rag_pipeline import load_resources, ask

# Archivos
GOLDEN_SET_FILE = "golden_set.json"
REPORT_FILE     = "evaluation_report.csv"


def load_golden_set() -> list[dict]:
    """Carga el conjunto de preguntas de prueba."""
    with open(GOLDEN_SET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(retrieved_sources: list[str], expected_source: str, k: int) -> float:
    """
    Precision@k: De los k fragmentos recuperados,
    ¿cuántos vienen de la fuente esperada?
    """
    top_k_sources = retrieved_sources[:k]
    relevant = sum(1 for s in top_k_sources if expected_source in s)
    return relevant / k


def recall_at_k(retrieved_sources: list[str], expected_source: str, k: int) -> float:
    """
    Recall@k: ¿Se encontró al menos un fragmento de la fuente esperada
    dentro de los primeros k resultados?
    """
    top_k_sources = retrieved_sources[:k]
    found = any(expected_source in s for s in top_k_sources)
    return 1.0 if found else 0.0


def evaluate(top_k: int = 5) -> pd.DataFrame:
    """Evalúa el sistema RAG con todas las preguntas del golden set."""

    print("=" * 55)
    print("  EVALUACIÓN DEL SISTEMA RAG - BMW")
    print("=" * 55)

    # Cargar recursos
    print("\n🔄 Cargando recursos...")
    chunks, index, model = load_resources()

    # Cargar golden set
    golden_set = load_golden_set()
    print(f"  Preguntas de prueba: {len(golden_set)}")
    print(f"  Top-k configurado:   {top_k}\n")

    results = []

    for item in golden_set:
        qid      = item["id"]
        question = item["question"]
        expected = item["expected_source"]

        print(f"  [{qid:02d}] {question[:60]}...")

        # Medir latencia
        start = time.time()
        result = ask(question, chunks, index, model, top_k=top_k)
        latency = round(time.time() - start, 2)

        # Obtener fuentes recuperadas
        retrieved_sources = [s["source"] for s in result["sources"]]

        # Calcular métricas
        p_at_k = precision_at_k(retrieved_sources, expected, top_k)
        r_at_k = recall_at_k(retrieved_sources, expected, top_k)

        results.append({
            "id":               qid,
            "question":         question,
            "expected_source":  expected,
            "retrieved_sources": ", ".join(retrieved_sources),
            "precision_at_k":   round(p_at_k, 4),
            "recall_at_k":      round(r_at_k, 4),
            "latency_seconds":  latency
        })

        print(f"       Precision@{top_k}: {p_at_k:.2f} | Recall@{top_k}: {r_at_k:.2f} | Latencia: {latency}s")

    # Crear DataFrame
    df = pd.DataFrame(results)

    # Promedios
    avg_precision = df["precision_at_k"].mean()
    avg_recall    = df["recall_at_k"].mean()
    avg_latency   = df["latency_seconds"].mean()

    print("\n" + "=" * 55)
    print("  RESULTADOS FINALES")
    print("=" * 55)
    print(f"  Precision@{top_k} promedio: {avg_precision:.4f}")
    print(f"  Recall@{top_k} promedio:    {avg_recall:.4f}")
    print(f"  Latencia promedio:       {avg_latency:.2f}s")
    print("=" * 55)

    # Guardar reporte
    df.to_csv(REPORT_FILE, index=False)
    print(f"\n✅ Reporte guardado en: {REPORT_FILE}")

    return df


def experiment_chunk_size():
    """
    Experimento: Compara dos tamaños de chunk.
    Requiere regenerar el índice con diferentes configuraciones en ingest.py.
    """
    print("\n🔬 EXPERIMENTO: Tamaño de chunk")
    print("  Para comparar chunk_size=300 vs chunk_size=600:")
    print("  1. En ingest.py cambia CHUNK_SIZE = 300 y ejecuta: python ingest.py")
    print("  2. Luego ejecuta: python evaluate.py")
    print("  3. Guarda el reporte como evaluation_report_300.csv")
    print("  4. Repite con CHUNK_SIZE = 600")
    print("  5. Compara los resultados de ambos reportes\n")


def experiment_top_k():
    """
    Experimento: Compara dos valores de top-k.
    """
    print("\n🔬 EXPERIMENTO: Comparación de top-k")
    print("  Evaluando con k=3...")
    df_k3 = evaluate(top_k=3)

    print("\n  Evaluando con k=7...")
    df_k7 = evaluate(top_k=7)

    print("\n📊 COMPARACIÓN k=3 vs k=7:")
    print(f"  Precision@3: {df_k3['precision_at_k'].mean():.4f}  |  Precision@7: {df_k7['precision_at_k'].mean():.4f}")
    print(f"  Recall@3:    {df_k3['recall_at_k'].mean():.4f}  |  Recall@7:    {df_k7['recall_at_k'].mean():.4f}")
    print(f"  Latencia@3:  {df_k3['latency_seconds'].mean():.2f}s  |  Latencia@7:  {df_k7['latency_seconds'].mean():.2f}s")


if __name__ == "__main__":
    evaluate(top_k=5)


