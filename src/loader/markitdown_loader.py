"""
Simplified document loader using Microsoft MarkItDown for LLM-ready document processing.
This replaces the complex document_loader.py with a cleaner, more reliable approach.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

from markitdown import MarkItDown
import bleach

from src.config import config_manager


@dataclass
class DocumentMetadata:
    """Document metadata structure."""
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    created_at: datetime
    modified_at: datetime
    language: str
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    document_id: str = ""
    
    def __post_init__(self):
        """Generate document ID after initialization."""
        if not self.document_id:
            content = f"{self.file_path}{self.file_size}{self.modified_at}"
            self.document_id = hashlib.md5(content.encode()).hexdigest()


@dataclass
class DocumentChunk:
    """Document chunk with metadata for vector storage."""
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    heading: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    word_count: int = 0
    page_number: Optional[int] = None  # Added for compatibility with vector store
    
    def __post_init__(self):
        """Calculate word count and chunk ID."""
        self.word_count = len(self.content.split())


class MarkItDownLoader:
    """Simplified document loader using MarkItDown for reliable document processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.md = MarkItDown()
        
    def load_document(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """
        Load a document from file path and return clean markdown content with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (markdown_content, metadata)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Get file stats
        stat = path.stat()
        
        # Convert document to markdown using MarkItDown
        try:
            result = self.md.convert(str(path))
            content = result.text_content
            
            # Clean and sanitize the content
            content = self._clean_content(content)
            
            # Detect language
            language = self._detect_language(content)
            
            # Create metadata
            metadata = DocumentMetadata(
                file_path=str(path),
                file_name=path.name,
                file_size=stat.st_size,
                file_type=path.suffix,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                language=language,
                word_count=len(content.split())
            )
            
            self.logger.info(f"Loaded document: {path.name} ({metadata.word_count} words)")
            
            return content, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load document {file_path}: {e}")
            raise
    
    def chunk_document(self, content: str, metadata: DocumentMetadata, 
                      chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """
        Split markdown content into chunks with overlap for vector storage.
        
        Args:
            content: Markdown content
            metadata: Document metadata
            chunk_size: Maximum words per chunk
            overlap: Overlap words between chunks
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Split by markdown headings to preserve structure
        sections = self._split_by_markdown_headings(content)
        
        chunk_index = 0
        for section_heading, section_content in sections:
            # Split section into smaller chunks if needed
            words = section_content.split()
            
            start_idx = 0
            while start_idx < len(words):
                end_idx = min(start_idx + chunk_size, len(words))
                
                # Add overlap from previous chunk
                if start_idx > 0:
                    overlap_start = max(0, start_idx - overlap)
                    chunk_words = words[overlap_start:end_idx]
                else:
                    chunk_words = words[start_idx:end_idx]
                
                chunk_content = ' '.join(chunk_words)
                
                # Create chunk
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=metadata,
                    chunk_index=chunk_index,
                    heading=section_heading,
                    start_char=0,  # Could be calculated if needed
                    end_char=len(chunk_content)
                )
                
                chunks.append(chunk)
                chunk_index += 1
                
                # Move to next chunk with overlap
                start_idx = end_idx - overlap if end_idx < len(words) else end_idx
        
        self.logger.info(f"Created {len(chunks)} chunks from {metadata.file_name}")
        return chunks
    
    def _split_by_markdown_headings(self, content: str) -> List[Tuple[Optional[str], str]]:
        """Split content by markdown headings to preserve structure."""
        sections = []
        current_heading = None
        current_content = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Check if line is a markdown heading
            if line.startswith('#'):
                # Save previous section
                if current_content:
                    sections.append((current_heading, '\n'.join(current_content)))
                
                # Start new section
                current_heading = line.lstrip('#').strip()
                current_content = []
            else:
                if line:  # Skip empty lines
                    current_content.append(line)
        
        # Add final section
        if current_content:
            sections.append((current_heading, '\n'.join(current_content)))
        
        # If no headings found, return content as single section
        if not sections:
            sections = [(None, content)]
        
        return sections
    
    def _clean_content(self, content: str) -> str:
        """Clean and sanitize markdown content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Remove common boilerplate patterns
        boilerplate_patterns = [
            r'©\s*\d{4}.*?all rights reserved',
            r'confidential.*?distribution',
            r'this document.*?proprietary',
            r'page \d+ of \d+',
            r'printed on.*?\d{4}',
        ]
        
        for pattern in boilerplate_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove sensitive information patterns
        sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
        ]
        
        for pattern in sensitive_patterns:
            content = re.sub(pattern, '[REDACTED]', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection based on common words."""
        # Turkish common words
        turkish_words = ['ve', 'bir', 'bu', 'ile', 'için', 'olan', 'olarak', 'çok', 'daha', 'karbon']
        
        # English common words
        english_words = ['and', 'the', 'for', 'with', 'this', 'that', 'are', 'carbon', 'climate', 'policy']
        
        words = content.lower().split()
        
        turkish_count = sum(1 for word in words if word in turkish_words)
        english_count = sum(1 for word in words if word in english_words)
        
        return 'tr' if turkish_count > english_count else 'en'


class DocumentBatch:
    """Batch processing for multiple documents using MarkItDown."""
    
    def __init__(self, loader: MarkItDownLoader):
        self.loader = loader
        self.logger = logging.getLogger(__name__)
        
    def load_directory(self, directory_path: str) -> List[Tuple[str, DocumentMetadata]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of (content, metadata) tuples
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Supported file extensions
        supported_extensions = {'.docx', '.pdf', '.pptx', '.md', '.txt'}
        
        documents = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    content, metadata = self.loader.load_document(str(file_path))
                    documents.append((content, metadata))
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents