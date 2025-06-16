import chromadb
from chromadb.config import Settings
import os
import zipfile
from bs4 import BeautifulSoup
import tempfile
from tqdm import tqdm
import re
from datetime import datetime
from api_embeddings.semantic_chunker import BlenderDocChunker
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Mapping
import psutil
import tarfile  # Add this import

class EmbeddingsManager:
    def __init__(self, persist_directory="./chroma_db", chunk_size=1000, chunk_overlap=200, 
                 embedding_model="all-MiniLM-L6-v2", batch_size=50):
        """Initialize the embeddings manager with a persistent storage location."""
        
        # Auto-extract database if needed (for Blender add-on distribution)
        self._extract_database_if_needed()
        
        self.batch_size = batch_size
        self.persist_directory = os.path.abspath(persist_directory)
        print(f"Initializing ChromaDB client with directory: {self.persist_directory}")
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the client with settings for the latest Chroma version
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Initialize ChromaDB's built-in embedding function
        import chromadb.utils.embedding_functions as embedding_functions
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()  # type: ignore
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print("ChromaDB client initialized successfully")

    def _extract_database_if_needed(self):
        """Extract compressed ChromaDB for Blender add-on distribution."""
        addon_dir = os.path.dirname(__file__)
        chroma_path = os.path.join(addon_dir, "chroma_db")
        archive_path = os.path.join(addon_dir, "chroma_db.tar.gz")
        
        # If database doesn't exist but archive does, extract it
        if not os.path.exists(chroma_path) and os.path.exists(archive_path):
            print("Extracting Blender documentation database...")
            try:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(addon_dir)
                print("Database extracted successfully!")
                return True
            except Exception as e:
                print(f"Error extracting database: {e}")
                return False
        
        return os.path.exists(chroma_path)

    def ensure_database_ready(self):
        """Ensure database exists, rebuild if necessary."""
        if not self.collection_exists("blender_4.4_docs"):
            zip_path = os.path.join(os.path.dirname(__file__), "doc", "blender_python_reference_4_2.zip")
            if os.path.exists(zip_path):
                print("Building Blender documentation database (first time setup)...")
                result = self.process_zip_file(zip_path, "blender_4.4_docs")
                # Fix: Check if result is not None before calling .get()
                if result is not None:
                    return result.get('status') == 'completed'
                else:
                    return False
            else:
                print("Warning: Blender documentation not found. Some features may not work.")
                return False
        return True

    def clear_database(self):
        """Clear the database with improved error handling and verification."""
        try:
            if self.client:
                collections = self.client.list_collections()
                for collection in collections:
                    self.client.delete_collection(collection.name)
                print("All collections successfully deleted.")
            else:
                print("Client is not initialized.")
        except Exception as e:
            print(f"Error clearing database: {str(e)}")

    def chunk_text(self, text, metadata):
        """Dummy implementation for chunk_text to resolve errors."""
        return [{'content': text, 'metadata': metadata}]

    def reduce_whitespace(self, text):
        """Clean and normalize text content."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

    def process_html_file(self, html_content, file_path):
        """Process an HTML file and extract relevant text content."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Get text content
        text = soup.get_text(separator=' ', strip=True)

        # Clean the text
        text = self.reduce_whitespace(text)

        # Create base metadata
        metadata = {
            'source': file_path,
            'type': 'blender_python_reference',
            'title': soup.title.string if soup.title and soup.title.string else "Untitled"
        }

        # Split into chunks
        chunks = self.chunk_text(text, metadata)

        return chunks

    def process_zip_file(self, zip_path: str, collection_name: str):
        """Process all HTML files in a zip archive with semantic chunking.
        
        Args:
            zip_path: Path to the zip file containing Blender documentation
            collection_name: Name of the ChromaDB collection to store embeddings
        """
        print(f"Starting processing of {zip_path}...")
        start_time = datetime.now()
        
        # Clear existing database
        print("Clearing existing database...")
        self.clear_database()
        
        # Initialize semantic chunker with optimal settings for Blender docs
        chunker = BlenderDocChunker(
            min_chunk_size=150,  # Slightly larger min size for better context
            max_chunk_size=1200,  # Increased max size for more complete code examples
            overlap=150  # Increased overlap for better context between chunks
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Extracting {zip_path} to temporary directory...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                print("Extraction completed successfully.")
            except Exception as e:
                print(f"Error extracting zip file: {str(e)}")
                return
            
            # Get list of HTML files
            print("Scanning for HTML files...")
            html_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.html') and not file.startswith('.'):
                        html_files.append(os.path.join(root, file))
            
            if not html_files:
                print("No HTML files found in the zip archive.")
                return
                
            print(f"Found {len(html_files)} HTML files to process...")
            
            # Process all HTML files with progress bar
            all_chunks = []
            processed_count = 0
            error_count = 0
            
            print("Processing HTML files and extracting semantic chunks...")
            for file_path in tqdm(html_files, desc="Processing files", unit="file"):
                relative_path = os.path.relpath(file_path, temp_dir)
                file_name = os.path.basename(file_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        html_content = f.read()
                    
                    # Extract basic metadata from file path and content
                    soup = BeautifulSoup(html_content, 'html.parser')
                    title = soup.title.string if soup.title else "Untitled"
                    
                    # Get the first h1 as the main section
                    h1 = soup.find('h1')
                    section = h1.get_text().strip() if h1 else ""
                    
                    # Get all h2s for subsections
                    h2s = [h2.get_text().strip() for h2 in soup.find_all('h2')]
                    
                    # Get all code blocks for better context
                    code_blocks = [code.get_text().strip() for code in soup.find_all(['pre', 'code'])]
                    
                    # Base metadata for all chunks from this file
                    base_metadata = {
                        'source': relative_path,
                        'file_name': file_name,
                        'title': title[:200] if title else "Untitled",  # Truncate long titles or use default
                        'section': section[:200],
                        'subsections': h2s[:5],  # Limit to first 5 subsections
                        'has_code': len(code_blocks) > 0,
                        'code_block_count': len(code_blocks),
                        'processing_time': datetime.now().isoformat(),
                        'blender_version': '4.4',
                        'content_type': 'api_reference' if 'reference' in relative_path.lower() else 'documentation'
                    }
                    
                    # Use semantic chunking
                    chunks = chunker.chunk_html(html_content, relative_path)
                    
                    # Enhance each chunk with additional metadata
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = {
                            **base_metadata,
                            'chunk_id': f"{file_name}_{i:04d}",
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'chunk_type': chunk['metadata'].get('chunk_type', 'text'),
                            'element_type': chunk['metadata'].get('element_type', '')
                        }
                        chunk['metadata'] = chunk_metadata
                    
                    all_chunks.extend(chunks)
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"\nError processing {file_path}: {str(e)}")
                    continue
            
            # Process chunks and create embeddings
            if all_chunks:
                print(f"\nProcessing completed with {processed_count} files successfully processed and {error_count} errors.")
                print(f"Generated {len(all_chunks)} semantic chunks in total.")
                
                # Sort chunks by source and chunk index for better organization
                all_chunks.sort(key=lambda x: (x['metadata']['source'], x['metadata']['chunk_index']))
                
                # Add chunks to collection in batches
                print("\nGenerating embeddings and storing in ChromaDB...")
                
                # First, ensure we can create the collection
                collection = self.get_or_create_collection(collection_name)
                if collection is None:
                    print("Failed to create collection. Check error messages above.")
                    return {
                        'status': 'failed',
                        'error': 'Failed to create collection',
                        'files_processed': len(html_files),
                        'successful_files': processed_count,
                        'failed_files': error_count,
                        'total_chunks': len(all_chunks)
                    }
                
                # Prepare all data first
                texts = []
                metadatas = []
                ids = []
                
                for chunk in all_chunks:
                    texts.append(chunk['content'])
                    metadatas.append(chunk['metadata'])
                    ids.append(chunk['metadata']['chunk_id'])
                
                # Now add all texts using the improved add_texts method
                success = self.add_texts(collection_name, texts, metadatas, ids)
                
                if not success:
                    print("Warning: Some chunks may not have been added successfully.")
                
                # Refresh the collection reference
                collection = self.get_or_create_collection(collection_name)
                
                # Print summary
                end_time = datetime.now()
                duration = end_time - start_time
                
                print("\n" + "="*60)
                print("Processing Complete!")
                print("-"*60)
                print(f"Total files processed: {len(html_files)}")
                print(f"Files successfully processed: {processed_count}")
                print(f"Files with errors: {error_count}")
                print(f"Total chunks created: {len(all_chunks)}")
                print(f"Processing time: {duration}")
                print("="*60)
                    
                return {
                    'status': 'completed',
                    'files_processed': len(html_files),
                    'successful_files': processed_count,
                    'failed_files': error_count,
                    'total_chunks': len(all_chunks),
                    'processing_time_seconds': duration.total_seconds(),
                    'collection_name': collection_name,
                }
            else:
                print("\nNo valid content chunks were generated from the HTML files.")
                return {
                    'status': 'failed',
                    'error': 'No valid content chunks generated',
                    'files_processed': len(html_files)
                }


    def collection_exists(self, collection_name):
        """Check if a collection exists."""
        try:
            self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function  # type: ignore
            )
            return True
        except Exception:
            return False
        
    def get_or_create_collection(self, collection_name):
        """Get an existing collection or create a new one if it doesn't exist."""
        embedding_dimensionality = 384  # all-minilm-l6-v2-f32 uses 384 dimensions
        try:
            # Attempt to get the collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function  # type: ignore
            )
            
            # Validate collection metadata
            if not collection.metadata or collection.metadata.get("dimensionality") != embedding_dimensionality:
                print(f"Collection dimensionality mismatch or missing metadata. Recreating collection '{collection_name}'...")
                
                # Delete the collection
                self.client.delete_collection(collection_name)
                
                # Clear persistent storage directory if necessary
                persist_path = os.path.join(self.persist_directory, collection_name)
                if os.path.exists(persist_path):
                    shutil.rmtree(persist_path)
                    print(f"Cleared persistent storage for collection '{collection_name}'.")
                
                # Recreate the collection
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,  # type: ignore
                    metadata={"dimensionality": embedding_dimensionality}
                )
            return collection
        except ValueError:
            # Create collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,  # type: ignore
                metadata={"dimensionality": embedding_dimensionality}
            )
            return collection
        except Exception as e:
            print(f"Error getting or creating collection: {str(e)}")
            return None

    def add_texts(self, collection_name, texts, metadatas=None, ids=None):
        """Add texts to the collection with optimized batch processing."""
        if not texts:
            print("No texts provided to add to collection.")
            return False

        if metadatas is None:
            metadatas = [{}] * len(texts)
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]

        # Validate input lengths
        if len(texts) != len(metadatas) or len(texts) != len(ids):
            print("Error: texts, metadatas, and ids must have the same length")
            return False

        # Get or create the collection
        collection = self.get_or_create_collection(collection_name)
        if collection is None:
            print("Failed to get or create collection")
            return False

        # Start with the default batch size
        batch_size = self.batch_size
        print(f"Processing {len(texts)} texts with initial batch size of {batch_size}...")

        from concurrent.futures import ThreadPoolExecutor
        success = True

        def process_batch(start_idx):
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_metadatas = metadatas[start_idx:start_idx + batch_size]
            batch_ids = ids[start_idx:start_idx + batch_size]

            try:
                # Clean metadata
                cleaned_metadatas: Sequence[Mapping[str, str | int | float | bool]] = [
                    {k: v for k, v in meta.items() if isinstance(k, str)}
                    for meta in batch_metadatas
                ]

                collection.add(
                    documents=batch_texts,
                    metadatas=[{k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))} for metadata in cleaned_metadatas],
                    ids=batch_ids
                )
            except Exception as e:
                print(f"Error adding batch starting at index {start_idx}: {str(e)}")
                return False

            return True

        with ThreadPoolExecutor() as executor:
            for i in range(0, len(texts), batch_size):
                # Monitor system memory usage
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 80:  # If memory usage exceeds 80%
                    batch_size = max(10, batch_size // 2)  # Reduce batch size
                    print(f"High memory usage detected. Reducing batch size to {batch_size}.")
                elif memory_info.percent < 50 and batch_size < self.batch_size * 2:  # If memory usage is low
                    batch_size = min(self.batch_size * 2, batch_size * 2)  # Increase batch size
                    print(f"Low memory usage detected. Increasing batch size to {batch_size}.")

                future = executor.submit(process_batch, i)
                if not future.result():
                    success = False

        return success
                
    def query_collection(self, collection_name, query_texts, n_results=5):
        """Query the collection for similar texts."""
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                print(f"Failed to get collection: {collection_name}")
                return None

            # Use query_texts instead of query_embeddings to let ChromaDB handle embedding internally
            # This avoids dimension mismatch issues
            results = collection.query(query_texts=query_texts, n_results=n_results)

            # Handle results safely
            formatted_results = []
            documents_list = results.get('documents', [])
            metadatas_list = results.get('metadatas', [])
            distances_list = results.get('distances', [])
            ids_list = results.get('ids', [])

            for i, query in enumerate(query_texts):
                # Safe access for lists
                if documents_list and isinstance(documents_list, list):
                    documents = documents_list[i] if i < len(documents_list) else []
                else:
                    documents = []

                if metadatas_list and isinstance(metadatas_list, list):
                    metadatas = metadatas_list[i] if i < len(metadatas_list) else []
                else:
                    metadatas = []

                if distances_list and isinstance(distances_list, list):
                    distances = distances_list[i] if i < len(distances_list) else []
                else:
                    distances = []

                formatted_results.append({
                    'query': query,
                    'documents': documents,
                    'metadatas': metadatas,
                    'distances': distances,
                    'ids': ids_list[i] if i < len(ids_list) else []
                })

            return formatted_results
        except Exception as e:
            print(f"Error querying collection: {str(e)}")
            return None

    def get_collection_size(self, collection_name):
        """Get the size of the collection safely."""
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return 0

            # Use count() method if available, otherwise query for all documents
            try:
                return collection.count()
            except AttributeError:
                # Fallback: get all documents with a minimal query
                results = collection.get()
                documents = results.get('documents', [])
                return len(documents) if documents else 0
        except Exception as e:
            print(f"Error getting collection size: {str(e)}")
            return 0

# Example usage
if __name__ == "__main__":
    # Initialize the embeddings manager
    manager = EmbeddingsManager()
    
    # Process the Blender Python reference documentation
    zip_path = "doc/blender_python_reference_4_4.zip"
    if os.path.exists(zip_path):
        manager.process_zip_file(zip_path, "blender_4.4_docs")
        
        # Example query
        query = "How do I create a new mesh in Blender using Python?"
        results = manager.query_collection("blender_4.4_docs", [query])
        
        if results:
            print("\nQuery results:")
            # Use the formatted results structure
            for result in results:
                print(f"\nQuery: {result['query']}")
                for i, (doc, metadata) in enumerate(zip(result['documents'], result['metadatas'])):
                    print(f"\nResult {i+1}:")
                    print(f"Source: {metadata.get('source', 'Unknown')}")
                    print(f"Text: {doc[:200]}...")  # Print first 200 chars
    else:
        print(f"Zip file not found: {zip_path}")