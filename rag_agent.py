# rag_agent.py

import os
import json
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

try:
    import openai
except ImportError:
    openai = None
try:
    import requests
except ImportError:
    requests = None

# Set this before usage or load from secure config
def set_openai_key(api_key: str):
    if openai:
        openai.api_key = api_key

class RAGDocChunk:
    def __init__(self, content: str, embedding: List[float]):
        self.content = content
        self.embedding = embedding

class RAGRetriever:
    def __init__(self, chunks: List[RAGDocChunk]):
        self.chunks = chunks

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = [RAGDocChunk(d['content'], d['embedding']) for d in data]
        return cls(chunks)

    @classmethod
    def from_chromadb(cls, db_path: str):
        """
        Load document chunks and embeddings from a ChromaDB SQLite database.
        Assumes a table with columns for content and embedding (as BLOB or JSON).
        """
        import sqlite3
        import json
        chunks = []
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            # Try to find the table and columns (adjust as needed for your schema)
            # ChromaDB default: collection, embedding, document, id, etc.
            # We'll look for 'embeddings' and 'documents' tables
            # Try to get embeddings and documents from the default ChromaDB schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            if 'embeddings' in tables and 'documents' in tables:
                # Join embeddings and documents on document_id
                cursor.execute('''
                    SELECT documents.content, embeddings.embedding
                    FROM embeddings
                    JOIN documents ON embeddings.document_id = documents.id
                ''')
                rows = cursor.fetchall()
                for content, embedding_blob in rows:
                    # Embedding may be stored as JSON or BLOB
                    try:
                        embedding = json.loads(embedding_blob)
                    except Exception:
                        # If not JSON, try to decode as bytes (float32 array)
                        import numpy as np
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
                    chunks.append(RAGDocChunk(content, embedding))
            else:
                raise Exception("ChromaDB schema not recognized. Expected 'embeddings' and 'documents' tables.")
        finally:
            conn.close()
        return cls(chunks)

    def retrieve(self, query_embedding: List[float], top_k=4) -> List[str]:
        vectors = np.array([chunk.embedding for chunk in self.chunks])
        similarities = cosine_similarity(np.array([query_embedding]), vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[i].content for i in top_indices]

def embed_texts(texts: List[str], provider: str = 'openai', model: Optional[str] = None, openai_key: Optional[str] = None) -> List[List[float]]:
    if provider == 'openai':
        if openai is None:
            raise ImportError('openai package not installed')
        if openai_key:
            openai.api_key = openai_key
        response = openai.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    elif provider == 'ollama':
        if requests is None:
            raise ImportError('requests package not installed')
        # Use Ollama's /api/embeddings endpoint
        url = 'http://localhost:11434/api/embeddings'
        payload = {"model": model or "nomic-embed-text", "prompt": texts}
        r = requests.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["embeddings"]
    else:
        raise ValueError(f"Unknown provider: {provider}")

def create_embedding_json(doc_path: str, output_path: str, chunk_size: int = 500, provider: str = 'openai', model: Optional[str] = None, openai_key: Optional[str] = None):
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = embed_texts(chunks, provider=provider, model=model, openai_key=openai_key)
    data = [{'content': c, 'embedding': e} for c, e in zip(chunks, embeddings)]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def query_with_rag(prompt: str, retriever: RAGRetriever, provider: str = 'openai', model: Optional[str] = None, openai_key: Optional[str] = None) -> str:
    embedded_query = embed_texts([prompt], provider=provider, model=model, openai_key=openai_key)[0]
    top_chunks = retriever.retrieve(embedded_query)
    context = "\n\n".join(top_chunks)
    messages = [
        {"role": "system", "content": "You are an expert Blender scripting assistant."},
        {"role": "user", "content": f"Relevant context:\n{context}\n\nPrompt:\n{prompt}"}
    ]
    if provider == 'openai':
        if openai is None:
            raise ImportError('openai package not installed')
        if openai_key:
            openai.api_key = openai_key
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,  # type: ignore
            temperature=0.3
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
    elif provider == 'ollama':
        if requests is None:
            raise ImportError('requests package not installed')
        url = 'http://localhost:11434/api/chat'
        payload = {"model": model or "llama3", "messages": messages}
        r = requests.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        # Ollama returns {"message": {"role":..., "content":...}}
        return data["message"]["content"]
    else:
        raise ValueError(f"Unknown provider: {provider}")
