import os
import warnings
import pandas as pd
import ast

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ── Tuning knobs ──────────────────────────────────────────────────────────────
# How many retrieved chunks to keep per sample when scoring.
# 25 chunks → ~14k chars, fills the entire LLM context and serialises inference.
# 5 chunks keeps the highest-ranked evidence while cutting context by ~80%.
MAX_CONTEXT_CHUNKS = 5

# Hard cap (chars) per individual chunk to avoid one giant table poisoning scores.
MAX_CHUNK_CHARS = 1200
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("--- Setting up RAGAS with Local Llama 3.1 8B Judge ---")

    # 1. Cấu hình Judge
    # Llama 3.1 follows English instructions reliably with Vietnamese content.
    # JSON mode is kept to improve schema adherence for RAGAS structured outputs.
    local_judge = ChatOllama(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        format="json",
        temperature=0,
        num_ctx=8192,
        timeout=300,
        system=(
            "You are a strict JSON evaluation engine. "
            "Return only valid JSON with no markdown, no prose, and no code fences."
        ),
    )

    # 2. Cấu hình Embedding local
    local_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # 3. Khởi tạo Metrics với LLM đã chọn
    metrics = [
        Faithfulness(llm=local_judge),
        AnswerRelevancy(llm=local_judge, embeddings=local_embeddings),
        ContextPrecision(llm=local_judge),
        ContextRecall(llm=local_judge),
    ]

    # 4. Load dữ liệu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(script_dir), "ragas_dataset.csv")
    try:
        df = pd.read_csv(csv_path)
        df['contexts'] = df['contexts'].apply(ast.literal_eval)

        # ── Context trimming ──────────────────────────────────────────────────
        # Keep only the top-MAX_CONTEXT_CHUNKS chunks (already ranked by RRF score
        # in generate_ragas_data, so first = most relevant).
        # Also hard-cap each chunk to MAX_CHUNK_CHARS to avoid runaway tables.
        def trim_contexts(ctx_list):
            trimmed = ctx_list[:MAX_CONTEXT_CHUNKS]
            return [c[:MAX_CHUNK_CHARS] for c in trimmed]

        df['contexts'] = df['contexts'].apply(trim_contexts)
        # ─────────────────────────────────────────────────────────────────────

        dataset = Dataset.from_pandas(df)
        avg_chars = df['contexts'].apply(lambda c: sum(len(x) for x in c)).mean()
        print(f"Loaded {len(dataset)} samples. Avg context size after trim: {avg_chars:.0f} chars.")
    except Exception as e:
        print(f" Error loading CSV: {e}")
        return

    print("\nRunning RAGAS Evaluation LOCALLY...")
    print(f"Judge: llama3.1:8b (json mode) | Context: top {MAX_CONTEXT_CHUNKS} chunks, max {MAX_CHUNK_CHARS} chars each | workers=1 | num_ctx=8192")

    run_config = RunConfig(
        max_workers=1,
        max_wait=420,    # 7 min timeout per job (Llama 3.1 is slightly slower than Qwen)
        max_retries=8,   # More retries to handle occasional malformed LLM output
    )

    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=local_judge,
            embeddings=local_embeddings,
            run_config=run_config,
        )
    except Exception as e:
        print(f"\n Evaluation failed: {e}")
        return

    print("\n---  Evaluation Results (Local) ---")
    print(results)

    output_file = "ragas_results_local.csv"
    results.to_pandas().to_csv(output_file, index=False)
    print(f"\n Detailed report saved to {output_file}")

if __name__ == "__main__":
    run_evaluation()