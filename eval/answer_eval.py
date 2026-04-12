from groq import Groq


def score_faithfulness(context, answer, api_key):
    client = Groq(api_key=api_key)
    prompt = (
        "You are an evaluation assistant. Rate how faithfully the answer is supported by the context. "
        "Return ONLY a float between 0.0 and 1.0 where 1.0 means fully faithful and 0.0 means not faithful at all.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score:"
    )
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        text = response.choices[0].message.content.strip()
        return float(text)
    except Exception:
        return -1.0


def score_relevancy(question, answer, api_key):
    client = Groq(api_key=api_key)
    prompt = (
        "You are an evaluation assistant. Rate how relevant the answer is to the question. "
        "Return ONLY a float between 0.0 and 1.0 where 1.0 means perfectly relevant and 0.0 means not relevant at all.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score:"
    )
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        text = response.choices[0].message.content.strip()
        return float(text)
    except Exception:
        return -1.0


def run_answer_eval(qa_pairs, api_key):
    per_question = []
    for pair in qa_pairs:
        faithfulness = score_faithfulness(pair["context"], pair["answer"], api_key)
        relevancy = score_relevancy(pair["question"], pair["answer"], api_key)
        per_question.append({
            "question": pair["question"],
            "faithfulness": faithfulness,
            "relevancy": relevancy,
        })

    valid_f = [q["faithfulness"] for q in per_question if q["faithfulness"] >= 0]
    valid_r = [q["relevancy"] for q in per_question if q["relevancy"] >= 0]

    return {
        "per_question": per_question,
        "avg_faithfulness": sum(valid_f) / len(valid_f) if valid_f else -1.0,
        "avg_relevancy": sum(valid_r) / len(valid_r) if valid_r else -1.0,
    }
