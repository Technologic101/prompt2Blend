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

    def retrieve(self, query_embedding: List[float], top_k=8, similarity_threshold=0.2, max_tokens=4000) -> List[str]:
        """
        Enhanced retrieval function for Blender API documentation.
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Maximum number of chunks to retrieve (default: 8)
            similarity_threshold: Minimum similarity score to include a chunk (default: 0.2)
            max_tokens: Approximate maximum number of tokens to include in result (default: 4000)
            
        Returns:
            List of relevant document chunks
        """
        vectors = np.array([chunk.embedding for chunk in self.chunks])
        similarities = cosine_similarity(np.array([query_embedding]), vectors)[0]
        
        # Get indices sorted by similarity (highest first)
        ranked_indices = similarities.argsort()[::-1]
        
        selected_chunks = []
        total_tokens = 0
        
        # Keep adding chunks until we hit token limit or run out of sufficiently similar chunks
        for idx in ranked_indices:
            # Stop if similarity falls below threshold
            if similarities[idx] < similarity_threshold:
                break
                
            chunk_content = self.chunks[idx].content
            # Approximate token count (rough estimate: 4 chars ≈ 1 token)
            chunk_tokens = len(chunk_content) // 4
            
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk_content)
                total_tokens += chunk_tokens
                
                # If we have enough chunks and tokens, stop
                if len(selected_chunks) >= top_k:
                    break
            else:
                # If this chunk would exceed our token budget, skip it
                continue
                
        # Ensure we return at least one chunk, even if it's below threshold
        if not selected_chunks and len(ranked_indices) > 0:
            selected_chunks.append(self.chunks[ranked_indices[0]].content)
            
        return selected_chunks
        
    def retrieve_with_diversity(self, query_embedding: List[float], top_k=8, diversity_factor=0.3) -> List[str]:
        """
        Retrieve chunks with diversity to cover more aspects of the Blender API.
        This is helpful for complex modeling tasks that might require multiple API references.
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Maximum number of chunks to retrieve
            diversity_factor: How much to prioritize diversity (0.0-1.0)
            
        Returns:
            List of diverse document chunks relevant to the query
        """
        vectors = np.array([chunk.embedding for chunk in self.chunks])
        similarities = cosine_similarity(np.array([query_embedding]), vectors)[0]
        
        result_chunks = []
        selected_indices = set()
        
        # Get first chunk (most similar)
        best_idx = similarities.argmax()
        result_chunks.append(self.chunks[best_idx].content)
        selected_indices.add(best_idx)
        
        # Iteratively select diverse chunks
        while len(result_chunks) < top_k:
            next_idx = -1
            max_score = -float('inf')
            
            for i in range(len(similarities)):
                if i in selected_indices:
                    continue
                    
                # Balance between similarity to query and diversity from selected chunks
                similarity_score = similarities[i]
                
                # Calculate diversity term (average distance from selected chunks)
                diversity = 0
                if selected_indices:
                    for selected_idx in selected_indices:
                        # Lower similarity between chunks = more diverse
                        chunk_similarity = cosine_similarity(
                            vectors[i].reshape(1, -1), vectors[selected_idx].reshape(1, -1)
                        )[0][0]
                        diversity -= chunk_similarity / len(selected_indices)
                
                # Combined score balancing relevance and diversity
                score = (1 - diversity_factor) * similarity_score + diversity_factor * diversity
                
                if score > max_score:
                    max_score = score
                    next_idx = i
            
            if next_idx >= 0 and similarities[next_idx] > 0.1:  # Ensure minimal relevance
                result_chunks.append(self.chunks[next_idx].content)
                selected_indices.add(next_idx)
            else:
                break
        
        return result_chunks

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

def query_with_rag(prompt: str, retriever: RAGRetriever, provider: str = 'openai', 
                model: Optional[str] = None, openai_key: Optional[str] = None, 
                use_diversity: bool = True, top_k: int = 8, 
                max_tokens: int = 4000, diversity_factor: float = 0.3) -> str:
    """
    Enhanced query function using RAG for retrieving Blender API documentation.
    
    Args:
        prompt: The user's query
        retriever: The RAG retriever instance
        provider: The LLM provider (openai or ollama)
        model: The LLM model to use
        openai_key: OpenAI API key (if provider is openai)
        use_diversity: Whether to use diversity-aware retrieval (default: True)
        top_k: Maximum number of chunks to retrieve (default: 8)
        max_tokens: Maximum token budget for retrieved chunks (default: 4000)
        diversity_factor: How much to prioritize diversity (0.0-1.0, default: 0.3)
        
    Returns:
        Generated response from the LLM
    """
    embedded_query = embed_texts([prompt], provider=provider, model=model, openai_key=openai_key)[0]
    
    # Choose retrieval method based on parameters
    if use_diversity:
        top_chunks = retriever.retrieve_with_diversity(
            embedded_query, top_k=top_k, diversity_factor=diversity_factor
        )
    else:
        top_chunks = retriever.retrieve(
            embedded_query, top_k=top_k, max_tokens=max_tokens
        )
    
    # Add markers to help distinguish between different document chunks
    marked_chunks = [f"--- DOCUMENT CHUNK {i+1} ---\n{chunk}" for i, chunk in enumerate(top_chunks)]
    context = "\n\n".join(marked_chunks)
    system_prompt = """You are an expert Blender Python API assistant. 
Your task is to help users create Python scripts for Blender 4.4 based on the provided documentation.

IMPORTANT GUIDELINES:
1. Use only the Blender 4.4 Python API, referring to the provided context when necessary
2. Prioritize the most recent and relevant API information from the context
3. If multiple approaches are possible, use the most modern, efficient, and Pythonic approach
4. Do not include comments or any additional text in the generated code
5. If there are multiple document chunks with conflicting information, prioritize the newer documentation
6. Only generate code that would actually work in Blender 4.4

For code generation, include complete imports and proper error handling where appropriate.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is relevant documentation from the Blender 4.4 Python API:\n\n{context}\n\nBased on this documentation, please help me with the following request:\n\n{prompt}"}
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

def get_retriever(db_path: Optional[str] = None) -> Optional[RAGRetriever]:
    """
    Initialize and return a RAGRetriever instance optimized for Blender API documentation.
    
    The returned retriever has enhanced capabilities:
    - Standard retrieval: retriever.retrieve() - Gets the most similar chunks
    - Diversity-aware retrieval: retriever.retrieve_with_diversity() - Gets diverse but relevant chunks
    
    When using with query_with_rag(), you can customize retrieval with parameters:
    - use_diversity: Toggle between standard and diversity-aware retrieval
    - top_k: Adjust the number of chunks to retrieve (8 recommended for complex queries)
    - max_tokens: Control the total context size for retrieval
    - diversity_factor: Adjust the balance between similarity and diversity (0.0-1.0)
        
    Args:
        db_path: Optional custom path to the ChromaDB database
        
    Returns:
        A RAGRetriever instance or None if initialization fails
    """
    try:
        # Default path is in the chroma_db directory next to this file
        if db_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(module_dir)
            db_path = os.path.join(project_dir, 'chroma_db', 'chroma.sqlite3')
        
        # Check if the database file exists
        if not os.path.exists(db_path):
            print(f"ChromaDB database not found at {db_path}")
            return None
            
        # Load the retriever from ChromaDB
        retriever = RAGRetriever.from_chromadb(db_path)
        print(f"Successfully initialized RAG retriever with {len(retriever.chunks)} chunks")
        return retriever
    except Exception as e:
        print(f"Failed to initialize RAG retriever: {e}")
        return None
