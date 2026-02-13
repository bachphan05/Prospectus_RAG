import os
import pandas as pd
import ast
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

def run_evaluation():
    print("--- Setting up RAGAS with Local Qwen 2.5 Judge ---")

    # 1. Cấu hình Judge (Dùng Qwen 2.5 thay cho Llama 3)
    local_judge = ChatOllama(
        model="qwen2.5:7b", # Đổi sang qwen2.5 để xử lý JSON chuẩn hơn
        base_url="http://localhost:11434",
        format="json",
        temperature=0,
        num_ctx=16384, # Tăng context window để đọc được nhiều chunk tài chính
        timeout=300    # Tăng thời gian chờ lên 5 phút
    )

    # 2. Cấu hình Embedding local
    local_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # 3. Khởi tạo Metrics với LLM đã chọn
    metrics = [
        Faithfulness(llm=local_judge),
        AnswerRelevancy(llm=local_judge),
        ContextPrecision(llm=local_judge),
        ContextRecall(llm=local_judge),
    ]

    # 4. Load dữ liệu
    csv_path = "ragas_dataset.csv"
    try:
        df = pd.read_csv(csv_path)
        df['contexts'] = df['contexts'].apply(ast.literal_eval)
        dataset = Dataset.from_pandas(df)
        print(f"Loaded {len(dataset)} samples.")
    except Exception as e:
        print(f" Error loading CSV: {e}")
        return

  

    print("\nRunning RAGAS Evaluation LOCALLY...")
    print("Lưu ý: Đang chạy tuần tự 1-1 để tránh lỗi JSON. Sẽ mất khoảng 5-10 phút.")
    
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=local_judge,
            embeddings=local_embeddings,
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