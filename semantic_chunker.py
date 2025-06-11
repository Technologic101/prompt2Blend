import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup, Tag
import tiktoken
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    
class BlenderDocChunker:
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000, overlap: int = 100):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for more accurate chunk sizing."""
        return len(self.tokenizer.encode(text))
    
    def _split_at_heading(self, element: Tag, level: int) -> List[Tag]:
        """Recursively split content at headings of the specified level."""
        result = []
        current_section = []
        
        for child in element.children:
            if isinstance(child, Tag) and child.name and child.name.startswith('h') and child.name[1:].isdigit():
                heading_level = int(child.name[1:])
                if heading_level == level and current_section:
                    result.append(self._create_section_container(current_section))
                    current_section = []
            current_section.append(child)
            
        if current_section:
            result.append(self._create_section_container(current_section))
            
        return result
    
    def _create_section_container(self, elements: List[Any]) -> Tag:
        """Create a container for a section with proper HTML structure."""
        container = BeautifulSoup("<div class='section'></div>", 'html.parser').div
        if container is None:
            raise ValueError("Failed to initialize container")
        for el in elements:
            if isinstance(el, Tag):
                container.append(el)
        return container
    
    def _chunk_element(self, element: Tag, current_chunks: List[Chunk], metadata: Dict[str, Any]):
        """Recursively chunk HTML elements."""
        # Skip script, style, and other non-content elements
        if element.name in ['script', 'style', 'nav', 'footer', 'header']:
            return
            
        # If element is a code block or table, try to keep it whole
        if element.name in ['pre', 'code', 'table']:
            text = element.get_text(separator='\n', strip=True)
            if self._count_tokens(text) <= self.max_chunk_size:
                current_chunks.append(Chunk(
                    text=text,
                    metadata={
                        **metadata,
                        'element_type': element.name,
                        'chunk_type': 'code_block' if element.name in ['pre', 'code'] else 'table'
                    }
                ))
                return
        
        # Process child elements
        for child in element.children:
            if isinstance(child, Tag):
                self._chunk_element(child, current_chunks, metadata)
            elif isinstance(child, str) and child.strip():
                # Handle string elements
                pass
    
    def chunk_html(self, html_content: str, source: str) -> List[Dict]:
        """Chunk HTML content into semantic sections."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract metadata
        title = soup.title.text if soup.title else "Untitled"
        metadata = {
            'source': source,
            'title': title,
            'api_version': '4.2'  # Hardcoded as per your requirement
        }
        
        # First, split by main sections (h1)
        main_sections = self._split_at_heading(soup.body, 1) if soup.body else []
        
        chunks = []
        for section in main_sections:
            # For each main section, split by subsections (h2)
            subsections = self._split_at_heading(section, 2)
            
            for subsection in subsections:
                # For each subsection, create chunks
                section_chunks = []
                self._chunk_element(subsection, section_chunks, metadata)
                
                # Process the collected chunks
                current_chunk = []
                current_size = 0
                
                for chunk in section_chunks:
                    chunk_size = self._count_tokens(chunk.text)
                    
                    # If adding this chunk would exceed max size, finalize current chunk
                    if current_chunk and current_size + chunk_size > self.max_chunk_size:
                        chunks.append(Chunk(
                            text='\n\n'.join([c.text for c in current_chunk]),
                            metadata={
                                **current_chunk[0].metadata,
                                'chunk_type': 'combined'
                            }
                        ))
                        current_chunk = current_chunk[-self.overlap:]  # Keep overlap
                        current_size = sum(self._count_tokens(c.text) for c in current_chunk)
                    
                    current_chunk.append(chunk)
                    current_size += chunk_size
                
                # Add any remaining chunks
                if current_chunk:
                    chunks.append(Chunk(
                        text='\n\n'.join([c.text for c in current_chunk]),
                        metadata={
                            **current_chunk[0].metadata,
                            'chunk_type': 'combined'
                        }
                    ))
        
        return [{
            'content': chunk.text,
            'metadata': chunk.metadata
        } for chunk in chunks]