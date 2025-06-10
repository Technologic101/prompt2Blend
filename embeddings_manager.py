import chromadb
from chromadb.utils import embedding_functions
import os
import zipfile
from bs4 import BeautifulSoup
import ollama
import tempfile
import shutil
from tqdm import tqdm

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name="hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF"):
        self.model_name = model_name
        self.client = ollama.Client()
        
    def __call__(self, input):
        """Generate embeddings for the input texts.
        
        Args:
            input: A list of texts to generate embeddings for.
            
        Returns:
            A list of embeddings.
        """
        embeddings = []
        # Add progress bar for embedding generation
        for text in tqdm(input, desc="Generating embeddings", unit="text"):
            # Get embedding from Ollama
            response = self.client.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response['embedding'])
        return embeddings

class EmbeddingsManager:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize the embeddings manager with a persistent storage location."""
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Using Ollama with Qwen embeddings
        self.embedding_function = OllamaEmbeddingFunction()
        
    def collection_exists(self, collection_name):
        """Check if a collection exists."""
        try:
            self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            return True
        except Exception:
            return False
        
    def get_or_create_collection(self, collection_name):
        """Get an existing collection or create a new one if it doesn't exist."""
        try:
            if self.collection_exists(collection_name):
                print(f"Loading existing collection: {collection_name}")
                return self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            else:
                print(f"Creating new collection: {collection_name}")
                return self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
        except Exception as e:
            print(f"Error with collection {collection_name}: {e}")
            return None
            
    def add_texts(self, collection_name, texts, metadatas=None, ids=None):
        """Add texts to the collection with optional metadata and IDs."""
        collection = self.get_or_create_collection(collection_name)
            
        if collection:
            try:
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Added {len(texts)} texts to collection: {collection_name}")
            except Exception as e:
                print(f"Error adding texts: {e}")
                
    def query_collection(self, collection_name, query_texts, n_results=5):
        """Query the collection for similar texts."""
        if not self.collection_exists(collection_name):
            print(f"Collection {collection_name} does not exist")
            return None
            
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error querying collection: {e}")
            return None

    def process_html_file(self, html_content, file_path):
        """Process an HTML file and extract relevant text content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return {
            'text': text,
            'metadata': {
                'source': file_path,
                'type': 'blender_python_reference'
            }
        }

    def process_zip_file(self, zip_path, collection_name):
        """Process all HTML files in a zip archive and add them to the collection."""
        # Check if collection exists and has data
        if self.collection_exists(collection_name):
            collection = self.get_or_create_collection(collection_name)
            count = collection.count()
            if count > 0:
                print(f"Collection {collection_name} already exists with {count} documents")
                return
            
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip file
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Get list of HTML files
            html_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.html'):
                        html_files.append(os.path.join(root, file))
            
            # Process all HTML files with progress bar
            texts = []
            metadatas = []
            ids = []
            
            print(f"Processing {len(html_files)} HTML files...")
            for file_path in tqdm(html_files, desc="Processing HTML files", unit="file"):
                relative_path = os.path.relpath(file_path, temp_dir)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                processed = self.process_html_file(html_content, relative_path)
                
                texts.append(processed['text'])
                metadatas.append(processed['metadata'])
                ids.append(relative_path)
            
            # Add all processed texts to the collection
            if texts:
                print(f"Adding {len(texts)} processed files to collection...")
                self.add_texts(collection_name, texts, metadatas, ids)
                print(f"Successfully processed {len(texts)} HTML files from {zip_path}")
            else:
                print(f"No HTML files found in {zip_path}")

# Example usage
if __name__ == "__main__":
    # Initialize the embeddings manager
    manager = EmbeddingsManager()
    
    # Process the Blender Python reference documentation
    zip_path = "doc/blender_python_reference_4_2.zip"
    if os.path.exists(zip_path):
        manager.process_zip_file(zip_path, "blender_python_reference")
        
        # Example query
        query = "How do I create a new mesh in Blender using Python?"
        results = manager.query_collection("blender_python_reference", [query])
        
        if results:
            print("\nQuery results:")
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                print(f"\nSource: {metadata['source']}")
                print(f"Text: {doc[:200]}...")  # Print first 200 chars
    else:
        print(f"Zip file not found: {zip_path}") 