import fitz
import re
from typing import List
from utils import clean_text

def extract_text_from_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)

def extract_text_from_txt(file) -> str:
    return clean_text(file.read().decode("utf-8"))

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    sentences = split_into_sentences(text)
    chunks = []
    current_words = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_count + len(words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            overlap_words = current_words[-overlap:]
            current_words = overlap_words + words
            current_count = len(current_words)
        else:
            current_words.extend(words)
            current_count += len(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks

def process_file(file, chunk_size: int = 300, overlap: int = 50) -> List[dict]:
    name = file.name
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif name.endswith(".txt"):
        text = extract_text_from_txt(file)
    else:
        return []

    raw_chunks = chunk_text(text, chunk_size, overlap)
    chunks = []
    for i, chunk_text_item in enumerate(raw_chunks):
        chunks.append({
            "id": f"{name}_{i}",
            "text": chunk_text_item,
            "source_file": name,
            "chunk_index": i
        })
    return chunks