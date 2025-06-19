import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import pytest
import rag_agent

@pytest.fixture(scope="module")
def setup_openai_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in environment.")
    rag_agent.set_openai_key(api_key)


def test_embed_texts(setup_openai_key):
    texts = ["Blender is a 3D creation suite.", "Python scripting automates Blender."]
    embeddings = rag_agent.embed_texts(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(e, list) for e in embeddings)
    assert all(len(e) > 0 for e in embeddings)


def test_rag_retriever_retrieve():
    # Create fake chunks with simple embeddings
    chunks = [
        rag_agent.RAGDocChunk("foo", [1.0, 0.0]),
        rag_agent.RAGDocChunk("bar", [0.0, 1.0]),
        rag_agent.RAGDocChunk("baz", [0.5, 0.5]),
    ]
    retriever = rag_agent.RAGRetriever(chunks)
    # Query close to [1,0] should return "foo" first
    result = retriever.retrieve([0.9, 0.1], top_k=2)
    assert result[0] == "foo"
    assert len(result) == 2


def test_create_embedding_json_and_from_json(tmp_path, setup_openai_key):
    doc = "Blender is a 3D suite. Python scripting is powerful."
    doc_path = tmp_path / "doc.txt"
    out_path = tmp_path / "out.json"
    doc_path.write_text(doc)
    rag_agent.create_embedding_json(str(doc_path), str(out_path), chunk_size=20)
    assert out_path.exists()
    retriever = rag_agent.RAGRetriever.from_json(str(out_path))
    assert len(retriever.chunks) > 0
    assert isinstance(retriever.chunks[0].content, str)
    assert isinstance(retriever.chunks[0].embedding, list)


def test_query_with_rag(tmp_path, setup_openai_key):
    doc = "Blender is a 3D suite. Python scripting is powerful."
    doc_path = tmp_path / "doc.txt"
    out_path = tmp_path / "out.json"
    doc_path.write_text(doc)
    rag_agent.create_embedding_json(str(doc_path), str(out_path), chunk_size=20)
    retriever = rag_agent.RAGRetriever.from_json(str(out_path))
    prompt = "What is Blender?"
    answer = rag_agent.query_with_rag(prompt, retriever)
    assert isinstance(answer, str)
    assert len(answer) > 0
