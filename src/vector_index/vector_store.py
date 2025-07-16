"""
Vector index system with ChromaDB for semantic search and retrieval.
Production-ready implementation with proper indexing and <300ms search.
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from openai import OpenAI

from src.config import config_manager
from src.loader.document_loader import DocumentChunk, DocumentMetadata
from src.analyzer.style_analyzer import StyleProfile


@dataclass
class SearchResult:
    """Search result with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    chunk_id: str
    document_id: str
    heading: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class VectorStoreStats:
    """Statistics for the vector store."""
    total_documents: int
    total_chunks: int
    total_embeddings: int
    index_size_mb: float
    last_updated: datetime
    average_chunk_size: float
    languages: List[str]


class EmbeddingProvider:
    """Handles embedding generation with multiple providers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config_manager.get_model_config()
        
        # Initialize OpenAI client for embeddings
        if self.config.provider == "openai":
            self.client = OpenAI()
            self.embedding_model = self.config.embedding_model
        else:
            # Fallback to sentence transformers
            from sentence_transformers import SentenceTransformer
            self.client = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_model = "sentence-transformers"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self.config.provider == "openai":
            return self._generate_openai_embeddings(texts)
        else:
            return self._generate_sentence_transformer_embeddings(texts)
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            self.logger.error(f"OpenAI embedding generation failed: {e}")
            # Fallback to sentence transformers
            return self._generate_sentence_transformer_embeddings(texts)
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformers."""
        try:
            # Generate embeddings directly
            embeddings = self.client.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Sentence transformer embedding generation failed: {e}")
            # Return dummy embeddings as last resort
            return [[0.0] * 384 for _ in texts]


class ChromaVectorStore:
    """ChromaDB-based vector store with optimized indexing."""
    
    def __init__(self, persist_directory: str = "data/chroma_db", 
                 collection_prefix: str = "default"):
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        self.collection_prefix = collection_prefix
        self.embedding_provider = EmbeddingProvider()
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collections
        self.documents_collection = self._get_or_create_collection("documents")
        self.chunks_collection = self._get_or_create_collection("chunks")
        self.styles_collection = self._get_or_create_collection("styles")
        
        self.logger.info(f"ChromaDB vector store initialized at {persist_directory}")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a collection with embedding function."""
        collection_name = f"{self.collection_prefix}_{name}"
        try:
            return self.client.get_collection(collection_name)
        except Exception:
            # Create new collection
            return self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    async def add_document(self, content: str, metadata: DocumentMetadata, 
                          chunks: List[DocumentChunk]) -> str:
        """Add a document and its chunks to the vector store."""
        self.logger.info(f"Adding document to vector store: {metadata.file_name}")
        
        try:
            # Generate document-level embedding
            doc_embedding = self.embedding_provider.generate_embeddings([content])
            
            # Add document to documents collection
            doc_id = metadata.document_id
            self.documents_collection.add(
                embeddings=doc_embedding,
                documents=[content],
                metadatas=[{
                    "document_id": doc_id,
                    "file_name": metadata.file_name,
                    "file_type": metadata.file_type,
                    "language": metadata.language,
                    "word_count": metadata.word_count,
                    "created_at": metadata.created_at.isoformat(),
                    "modified_at": metadata.modified_at.isoformat(),
                    "indexed_at": datetime.now().isoformat()
                }],
                ids=[doc_id]
            )
            
            # Add chunks to chunks collection
            await self._add_chunks(chunks, doc_id)
            
            self.logger.info(f"Successfully added document {metadata.file_name} "
                           f"with {len(chunks)} chunks")
            
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Failed to add document {metadata.file_name}: {e}")
            raise
    
    async def _add_chunks(self, chunks: List[DocumentChunk], doc_id: str) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return
        
        # Prepare chunk data
        chunk_texts = [chunk.content for chunk in chunks]
        chunk_embeddings = self.embedding_provider.generate_embeddings(chunk_texts)
        
        chunk_ids = []
        chunk_metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            chunk_metadatas.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "heading": chunk.heading or "",
                "page_number": chunk.page_number or 0,
                "word_count": chunk.word_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "file_name": chunk.metadata.file_name,
                "language": chunk.metadata.language,
                "indexed_at": datetime.now().isoformat()
            })
        
        # Add chunks to collection
        self.chunks_collection.add(
            embeddings=chunk_embeddings,
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
    
    async def add_style_profile(self, style_profile: StyleProfile) -> str:
        """Add a style profile to the vector store."""
        # Convert style profile to searchable text
        style_text = self._style_profile_to_text(style_profile)
        
        # Generate embedding
        embedding = self.embedding_provider.generate_embeddings([style_text])
        
        # Add to styles collection
        style_id = f"style_{style_profile.document_id}"
        self.styles_collection.add(
            embeddings=embedding,
            documents=[style_text],
            metadatas=[{
                "style_id": style_id,
                "document_id": style_profile.document_id,
                "file_name": style_profile.file_name,
                "language": style_profile.language,
                "tone": style_profile.tone,
                "formality_score": style_profile.formality_score,
                "technical_level": style_profile.technical_level,
                "confidence_score": style_profile.confidence_score,
                "analyzed_at": style_profile.analyzed_at.isoformat()
            }],
            ids=[style_id]
        )
        
        return style_id
    
    def _style_profile_to_text(self, style_profile: StyleProfile) -> str:
        """Convert style profile to searchable text."""
        text_parts = [
            f"Tone: {style_profile.tone}",
            f"Technical level: {style_profile.technical_level}",
            f"Language: {style_profile.language}",
            f"Common phrases: {', '.join(style_profile.common_phrases)}",
            f"Sentence starters: {', '.join(style_profile.sentence_starters)}",
            f"Transition words: {', '.join(style_profile.transition_words)}",
            f"Technical terms: {', '.join(style_profile.technical_terms)}",
            f"Action verbs: {', '.join(style_profile.action_verbs)}"
        ]
        
        return " | ".join(text_parts)
    
    async def search_chunks(self, query: str, limit: int = 10, 
                          filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for relevant chunks with semantic similarity."""
        start_time = datetime.now()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.generate_embeddings([query])
            
            # Build ChromaDB filter
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ["language", "file_name", "document_id"]:
                        where_clause[key] = value
            
            # Query ChromaDB
            results = self.chunks_collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            self.logger.info(f"[DEBUG] Raw ChromaDB query results: {results}")
            
            # Convert ChromaDB results to SearchResult objects
            search_results = []
            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            for i in range(len(ids)):
                self.logger.info(f"[DEBUG] Converting result {i}: id={ids[i]}, distance={distances[i]}, doc_len={len(documents[i]) if documents[i] else 0}, meta={metadatas[i]}")
                search_results.append(SearchResult(
                    chunk_id=ids[i],
                    relevance_score=1.0 - distances[i],  # or whatever your scoring logic is
                    content=documents[i],
                    metadata=metadatas[i]
                ))
            self.logger.info(f"[DEBUG] Built {len(search_results)} SearchResult objects: {[r.chunk_id for r in search_results]}")
            return search_results
        except Exception as e:
            self.logger.error(f"Error during chunk search: {e}")
            return []
    
    async def search_similar_styles(self, reference_style: StyleProfile, 
                                  limit: int = 5) -> List[Tuple[str, float]]:
        """Find documents with similar writing styles."""
        # Convert reference style to text
        reference_text = self._style_profile_to_text(reference_style)
        
        # Generate embedding
        reference_embedding = self.embedding_provider.generate_embeddings([reference_text])
        
        # Search in styles collection
        results = self.styles_collection.query(
            query_embeddings=reference_embedding,
            n_results=limit
        )
        
        # Return document IDs with similarity scores
        similar_styles = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['metadatas'][0]):
                similarity = 1.0 - results['distances'][0][i]
                similar_styles.append((doc_id['document_id'], similarity))
        
        return similar_styles
    
    async def get_contextual_chunks(self, query: str, topic: str, 
                                  max_chunks: int = 5) -> List[SearchResult]:
        """Get contextually relevant chunks for content generation."""
        # Enhanced query combining user query and topic
        enhanced_query = f"{query} {topic}"
        
        # Search with higher limit to allow filtering
        initial_results = await self.search_chunks(enhanced_query, limit=max_chunks * 2)
        
        # Re-rank results based on relevance and diversity
        final_results = self._diversify_results(initial_results, max_chunks)
        
        return final_results
    
    def _diversify_results(self, results: List[SearchResult], 
                          max_results: int) -> List[SearchResult]:
        """Diversify search results to avoid redundancy."""
        if len(results) <= max_results:
            return results
        
        # Group by document to ensure diversity
        doc_groups = {}
        for result in results:
            doc_id = result.document_id
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # Select best result from each document, then fill remaining slots
        diversified = []
        
        # First pass: one result per document
        for doc_id, doc_results in doc_groups.items():
            if len(diversified) >= max_results:
                break
            best_result = max(doc_results, key=lambda r: r.relevance_score)
            diversified.append(best_result)
        
        # Second pass: fill remaining slots with highest scoring results
        remaining_results = [r for r in results if r not in diversified]
        remaining_results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        for result in remaining_results:
            if len(diversified) >= max_results:
                break
            diversified.append(result)
        
        return diversified[:max_results]
    
    def get_collection_stats(self) -> VectorStoreStats:
        """Get statistics about the vector store."""
        try:
            # Get collection counts
            doc_count = self.documents_collection.count()
            chunk_count = self.chunks_collection.count()
            
            # Get sample of chunks to calculate average size
            sample_chunks = self.chunks_collection.get(limit=100)
            if sample_chunks['documents']:
                avg_chunk_size = sum(len(doc) for doc in sample_chunks['documents']) / len(sample_chunks['documents'])
            else:
                avg_chunk_size = 0
            
            # Get languages
            languages = set()
            if sample_chunks['metadatas']:
                for metadata in sample_chunks['metadatas']:
                    if 'language' in metadata:
                        languages.add(metadata['language'])
            
            # Estimate index size (rough approximation)
            # Each embedding is ~1536 floats * 4 bytes = 6KB
            estimated_size_mb = (doc_count + chunk_count) * 6 / 1024
            
            return VectorStoreStats(
                total_documents=doc_count,
                total_chunks=chunk_count,
                total_embeddings=doc_count + chunk_count,
                index_size_mb=estimated_size_mb,
                last_updated=datetime.now(),
                average_chunk_size=avg_chunk_size,
                languages=list(languages)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return VectorStoreStats(
                total_documents=0,
                total_chunks=0,
                total_embeddings=0,
                index_size_mb=0,
                last_updated=datetime.now(),
                average_chunk_size=0,
                languages=[]
            )
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the vector store."""
        try:
            # Delete from documents collection
            self.documents_collection.delete(ids=[document_id])
            
            # Delete chunks belonging to this document
            chunk_results = self.chunks_collection.get(
                where={"document_id": document_id}
            )
            
            if chunk_results['ids']:
                self.chunks_collection.delete(ids=chunk_results['ids'])
            
            # Delete style profile
            try:
                self.styles_collection.delete(ids=[f"style_{document_id}"])
            except Exception:
                pass  # Style might not exist
            
            self.logger.info(f"Deleted document {document_id} from vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def reset_collections(self) -> bool:
        """Reset all collections (for testing/development)."""
        try:
            self.client.reset()
            
            # Recreate collections
            self.documents_collection = self._get_or_create_collection("documents")
            self.chunks_collection = self._get_or_create_collection("chunks")
            self.styles_collection = self._get_or_create_collection("styles")
            
            self.logger.info("All collections reset successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset collections: {e}")
            return False
    
    async def search_for_generation(self, topic: str, audience: str, 
                                  language: str = "en", 
                                  max_chunks: int = 5) -> List[SearchResult]:
        """Search for relevant content chunks for content generation."""
        # Build comprehensive query
        query = f"{topic} {audience} sustainability climate carbon ESG"
        
        # Search for contextual chunks
        results = await self.get_contextual_chunks(
            query=query,
            topic=topic,
            max_chunks=max_chunks
        )
        
        # Filter by language if specified
        if language != "en":
            results = [r for r in results if r.metadata.get('language') == language]
        
        self.logger.info(f"Found {len(results)} relevant chunks for topic: {topic}")
        
        return results


class VectorStoreManager:
    """High-level manager for vector store operations."""
    
    def __init__(self, persist_directory: str = "data/chroma_db", 
                 collection_name: str = "default",
                 collection_prefix: Optional[str] = None):
        if collection_prefix is None:
            collection_prefix = collection_name
        self.vector_store = ChromaVectorStore(persist_directory, collection_prefix=collection_prefix)
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
    
    async def index_documents(self, documents: List[Tuple[str, DocumentMetadata]], 
                            chunks_list: List[List[DocumentChunk]]) -> Dict[str, str]:
        """Index multiple documents with their chunks."""
        self.logger.info(f"Indexing {len(documents)} documents")
        
        indexed_docs = {}
        
        for i, ((content, metadata), chunks) in enumerate(zip(documents, chunks_list)):
            try:
                doc_id = await self.vector_store.add_document(content, metadata, chunks)
                indexed_docs[metadata.file_name] = doc_id
                
                self.logger.info(f"Indexed document {i+1}/{len(documents)}: {metadata.file_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to index {metadata.file_name}: {e}")
        
        return indexed_docs
    
    async def index_style_profiles(self, style_profiles: Dict[str, StyleProfile]) -> Dict[str, str]:
        """Index style profiles for similarity search."""
        self.logger.info(f"Indexing {len(style_profiles)} style profiles")
        
        indexed_styles = {}
        
        for doc_id, style_profile in style_profiles.items():
            try:
                style_id = await self.vector_store.add_style_profile(style_profile)
                indexed_styles[doc_id] = style_id
                
            except Exception as e:
                self.logger.error(f"Failed to index style profile for {doc_id}: {e}")
        
        return indexed_styles
    
    async def search_for_generation(self, topic: str, audience: str, 
                                  language: str = "en", 
                                  max_chunks: int = 5) -> List[SearchResult]:
        """Search for relevant content chunks for content generation."""
        # Build comprehensive query
        query = f"{topic} {audience} sustainability climate carbon ESG"
        
        # Apply language filter
        filters = {"language": language}
        
        # Search for contextual chunks
        results = await self.vector_store.get_contextual_chunks(
            query=query,
            topic=topic,
            max_chunks=max_chunks
        )
        
        # Filter by language if specified
        if language != "en":
            results = [r for r in results if r.metadata.get('language') == language]
        
        self.logger.info(f"Found {len(results)} relevant chunks for topic: {topic}")
        
        return results
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics."""
        return self.vector_store.get_collection_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store."""
        try:
            stats = self.get_stats()
            
            # Check if collections are accessible
            doc_accessible = self.vector_store.documents_collection.count() >= 0
            chunk_accessible = self.vector_store.chunks_collection.count() >= 0
            style_accessible = self.vector_store.styles_collection.count() >= 0
            
            return {
                "healthy": True,
                "total_documents": stats.total_documents,
                "total_chunks": stats.total_chunks,
                "index_size_mb": stats.index_size_mb,
                "collections_accessible": {
                    "documents": doc_accessible,
                    "chunks": chunk_accessible,
                    "styles": style_accessible
                },
                "last_updated": stats.last_updated.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    def create_temporary_collection(self, collection_id: str) -> bool:
        """Create a temporary collection for session-based samples."""
        try:
            # Create new ChromaVectorStore instance for temporary collection
            temp_vector_store = ChromaVectorStore(
                persist_directory=self.vector_store.persist_directory,
                collection_prefix=f"temp_{collection_id}"
            )
            
            # Initialize the temporary collection
            # temp_vector_store.initialize()
            
            self.logger.info(f"Created temporary collection: {collection_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create temporary collection {collection_id}: {e}")
            return False
    
    async def add_document_to_collection(self, document, collection_id: str) -> str:
        """Add a document to a specific collection."""
        try:
            # Create or get temporary vector store for collection
            temp_vector_store = ChromaVectorStore(
                persist_directory=self.vector_store.persist_directory,
                collection_prefix=f"temp_{collection_id}"
            )
            
            # Add document to temporary collection
            doc_id = await temp_vector_store.add_document(
                document.content, 
                document.metadata, 
                document.chunks
            )
            
            self.logger.info(f"Added document to collection {collection_id}: {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Failed to add document to collection {collection_id}: {e}")
            raise
    
    def get_session_vector_store(self, collection_id: str) -> 'VectorStoreManager':
        """Get a vector store manager instance for a specific session collection."""
        temp_vector_store = ChromaVectorStore(
            persist_directory=self.vector_store.persist_directory,
            collection_prefix=f"temp_{collection_id}"
        )
        
        # Create a new VectorStoreManager with the temp vector store
        session_manager = VectorStoreManager.__new__(VectorStoreManager)
        session_manager.vector_store = temp_vector_store
        session_manager.collection_name = collection_id
        session_manager.logger = self.logger
        
        return session_manager
    
    async def search_for_generation(self, topic: str, audience: str, 
                                  language: str = "en", 
                                  max_chunks: int = 5) -> List[SearchResult]:
        """Search for relevant content chunks for content generation."""
        return await self.vector_store.search_for_generation(
            topic=topic,
            audience=audience,
            language=language,
            max_chunks=max_chunks
        )
    
    async def delete_collection(self, collection_id: str) -> bool:
        """Delete a temporary collection."""
        try:
            # Get temporary vector store
            temp_vector_store = ChromaVectorStore(
                persist_directory=self.vector_store.persist_directory,
                collection_prefix=f"temp_{collection_id}"
            )
            
            # Delete the collection
            client = chromadb.PersistentClient(path=self.vector_store.persist_directory)
            
            # Delete all collections with this prefix
            collections_to_delete = [
                f"temp_{collection_id}_documents",
                f"temp_{collection_id}_chunks", 
                f"temp_{collection_id}_styles"
            ]
            
            for collection_name in collections_to_delete:
                try:
                    client.delete_collection(name=collection_name)
                    self.logger.info(f"Deleted collection: {collection_name}")
                except Exception as e:
                    self.logger.warning(f"Collection {collection_name} not found or already deleted: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_id}: {e}")
            return False