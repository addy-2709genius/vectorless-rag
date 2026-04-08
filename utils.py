import re
import string
from typing import List
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    STOPWORDS = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize('test')
except:
    nltk.download('wordnet')
    LEMMATIZER = WordNetLemmatizer()

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def truncate_context(chunks: List[dict], max_words: int = 6000) -> List[dict]:
    total = 0
    result = []
    for chunk in chunks:
        words = len(chunk["text"].split())
        if total + words > max_words:
            break
        result.append(chunk)
        total += words
    return result

def format_context(chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        header = f"[Chunk {i+1} | source: {chunk['source_file']} | chunk #{chunk['chunk_index']}]"
        parts.append(f"{header}\n\n{chunk['text']}\n\n---")
    return "\n".join(parts)