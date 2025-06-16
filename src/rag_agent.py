# rag_agent.py

import os
import json
import openai
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

# Set this before usage or load from secure config
def set_openai_key(api_key: str):
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

    def retrieve(self, query_embedding: List[float], top_k=4) -> List[str]:
        vectors = np.array([chunk.embedding for chunk in self.chunks])
        similarities = cosine_similarity(np.array([query_embedding]), vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[i].content for i in top_indices]

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Use the OpenAI API v1 for embeddings
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    # The response object has a .data attribute, which is a list of objects with an .embedding attribute
    return [item.embedding for item in response.data]

def create_embedding_json(doc_path: str, output_path: str, chunk_size: int = 500):
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = embed_texts(chunks)
    data = [{'content': c, 'embedding': e} for c, e in zip(chunks, embeddings)]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def query_with_rag(prompt: str, retriever: RAGRetriever) -> str:
    embedded_query = embed_texts([prompt])[0]
    top_chunks = retriever.retrieve(embedded_query)

    context = "\n\n".join(top_chunks)
    messages = [
        {"role": "system", "content": "You are an expert Blender scripting assistant."},
        {"role": "user", "content": f"Relevant context:\n{context}\n\nPrompt:\n{prompt}"}
    ]

    # Use the OpenAI API v1 for chat completions
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,  # type: ignore
        temperature=0.3
    )
    # The response object has a .choices attribute, which is a list of objects with a .message.content attribute
    content = response.choices[0].message.content
    return content if content is not None else ""
