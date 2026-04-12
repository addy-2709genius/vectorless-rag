from retrieval import build_index, search

TEST_SET = [
    {
        "query": "What is the main topic of the document?",
        "relevant_ids": ["doc.pdf_0", "doc.pdf_1"],  # TODO: replace with real chunk IDs from your document
    },
    {
        "query": "What are the key findings?",
        "relevant_ids": ["doc.pdf_2", "doc.pdf_3"],  # TODO: replace with real chunk IDs from your document
    },
    {
        "query": "What methodology was used?",
        "relevant_ids": ["doc.pdf_4", "doc.pdf_5"],  # TODO: replace with real chunk IDs from your document
    },
]


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return 0.0
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(top_k)


def run_retrieval_eval(chunks, index):
    per_query = []
    for test in TEST_SET:
        results = search(test["query"], index, chunks, top_k=20)
        retrieved_ids = [r["id"] for r in results]
        recall = recall_at_k(retrieved_ids, test["relevant_ids"], k=5)
        precision = precision_at_k(retrieved_ids, test["relevant_ids"], k=5)
        per_query.append({
            "query": test["query"],
            "recall_at_5": recall,
            "precision_at_5": precision,
        })

    avg_recall = sum(q["recall_at_5"] for q in per_query) / len(per_query) if per_query else 0.0
    avg_precision = sum(q["precision_at_5"] for q in per_query) / len(per_query) if per_query else 0.0

    return {
        "per_query": per_query,
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
    }
