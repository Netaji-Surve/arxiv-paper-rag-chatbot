import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.graph import execute_workflow
from src.rag.llm import llm


GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")


def load_ground_truth() -> dict:
    with open(GROUND_TRUTH_PATH) as f:
        data = json.load(f)
    return {item["question"]: item["reference"] for item in data}


def build_dataset() -> EvaluationDataset:
    ground_truth = load_ground_truth()
    samples = []

    for i, (question, reference) in enumerate(ground_truth.items(), 1):
        print(f"[{i}/{len(ground_truth)}] {question[:70]}...")
        result = execute_workflow(question)

        contexts = [doc.page_content for doc in result.get("documents", [])]

        samples.append(SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=contexts,
            reference=reference,
        ))

    return EvaluationDataset(samples=samples)


def main():
    print("Building evaluation dataset...\n")
    dataset = build_dataset()

    judge_llm = LangchainLLMWrapper(llm)
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithReference(),   # retriever: are retrieved chunks relevant?
        LLMContextRecall(),                   # retriever: was all needed info retrieved?
    ]

    run_config = RunConfig(timeout=240, max_workers=1)

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    print("\n=== RAGAS Results ===")
    print(results)

    df = results.to_pandas()
    output_path = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
