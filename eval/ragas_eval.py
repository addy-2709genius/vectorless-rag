def run_ragas_eval(qa_pairs):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
    except ImportError:
        return {"error": "RAGAS not installed. Run: pip install ragas datasets"}

    data = {
        "question": [p["question"] for p in qa_pairs],
        "answer": [p["answer"] for p in qa_pairs],
        "contexts": [p["contexts"] for p in qa_pairs],
        "ground_truth": [p["ground_truth"] for p in qa_pairs],
    }

    dataset = Dataset.from_dict(data)

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        return dict(result)
    except Exception as e:
        return {"error": str(e)}
