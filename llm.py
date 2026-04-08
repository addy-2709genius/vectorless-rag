import os
import time
import streamlit as st
from groq import Groq

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context chunks. Be direct and concise. Do not mention chunk numbers, chunk labels, or say things like 'according to Chunk 1'. Just answer naturally using the information. If the context does not contain enough information, say so clearly. Do not hallucinate or use outside knowledge."""

def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

def stream_answer(query: str, context: str, api_key: str, model: str, temperature: float):
    client = get_client(api_key)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        stream=True,
        temperature=temperature,
        max_tokens=1024,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.choices[0].delta.content or "" for chunk in stream
        )

    latency = round(time.time() - start, 2)
    return response, latency, model