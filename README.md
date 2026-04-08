# Vectorless RAG
> A product by Aaditya Raj Soni

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-22c55e?style=flat)
![BM25](https://img.shields.io/badge/Retrieval-BM25-22c55e?style=flat)

Document Q&A without any vector database or embedding model. BM25 retrieval + cross-encoder reranking + Groq LLM.

---

## System Architecture
PDF / TXT
↓
Chunker (300 words, 50 overlap)
↓
BM25 Index (NLTK tokenization)
↓
User Query → BM25 Retrieval → Top-20 Chunks
↓
Cross-Encoder Reranker
↓
Top-5 Chunks
↓
Groq LLM (streaming)
↓
Cited Answer
---

## Stack
- **Retrieval** — BM25 + NLTK tokenization
- **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM** — Groq (llama-3.3-70b, mixtral, gemma)
- **UI** — Streamlit

---

## Setup

```bash
git clone https://github.com/addy-2709genius/vectorless-rag.git
cd vectorless-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
echo "GROQ_API_KEY=gsk_..." > .env
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)
