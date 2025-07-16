"""
Session management for dynamic sample uploads.
Handles temporary storage and processing of user-uploaded samples.
"""

import os
import shutil
import tempfile
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager

from src.loader.markitdown_loader import MarkItDownLoader
from src.vector_index.vector_store import VectorStoreManager
from src.analyzer.style_analyzer import StyleProfileManager


@dataclass
class SessionSample:
    """Represents a sample file in a session."""
    file_path: Path
    original_name: str
    file_size: int
    processed: bool = False
    error: Optional[str] = None
    chunks_created: int = 0
    processing_time: float = 0.0


@dataclass
class ContentSession:
    """Represents a content generation session with temporary samples."""
    session_id: str
    created_at: datetime
    temp_dir: Path
    samples: List[SessionSample]
    vector_collection_id: str
    is_processed: bool = False
    cleanup_scheduled: bool = False
    
    def __post_init__(self):
        """Ensure temp directory exists."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)


class SessionManager:
    """Manages content generation sessions with temporary samples."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, 
                 style_manager: StyleProfileManager):
        self.vector_store = vector_store_manager
        self.style_manager = style_manager
        self.loader = MarkItDownLoader()
        self.logger = logging.getLogger(__name__)
        
        # Active sessions
        self.active_sessions: Dict[str, ContentSession] = {}
        
        # Session cleanup settings
        self.max_session_age = timedelta(hours=2)  # Sessions expire after 2 hours
        self.cleanup_interval = timedelta(minutes=30)  # Cleanup every 30 minutes
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the cleanup background task."""
        if self._cleanup_task is None:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
            except RuntimeError:
                # No event loop running, skip cleanup task for now
                self._cleanup_task = None
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                await self._cleanup_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    async def _cleanup_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if now - session.created_at > self.max_session_age:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.cleanup_session(session_id)
    
    def create_session(self) -> ContentSession:
        """Create a new content generation session."""
        session_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.mkdtemp(prefix=f"erguvan_session_{session_id}_"))
        
        # Create unique vector collection for this session
        vector_collection_id = f"session_{session_id}"
        
        session = ContentSession(
            session_id=session_id,
            created_at=datetime.now(),
            temp_dir=temp_dir,
            samples=[],
            vector_collection_id=vector_collection_id
        )
        
        self.active_sessions[session_id] = session
        
        self.logger.info(f"Created session {session_id} with temp dir: {temp_dir}")
        return session
    
    async def add_sample_to_session(self, session_id: str, file_content: bytes, 
                                   filename: str) -> SessionSample:
        """Add a sample file to a session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Save file to temp directory
        file_path = session.temp_dir / filename
        
        try:
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            sample = SessionSample(
                file_path=file_path,
                original_name=filename,
                file_size=len(file_content)
            )
            
            session.samples.append(sample)
            
            self.logger.info(f"Added sample {filename} to session {session_id}")
            return sample
            
        except Exception as e:
            self.logger.error(f"Failed to save sample {filename} to session {session_id}: {e}")
            raise
    
    async def process_session_samples(self, session_id: str) -> Dict[str, any]:
        """Process all samples in a session and create vector embeddings."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if session.is_processed:
            self.logger.info(f"Session {session_id} already processed")
            return self._get_processing_stats(session)
        
        self.logger.info(f"Processing {len(session.samples)} samples in session {session_id}")
        
        # Initialize temporary vector collection
        self.vector_store.create_temporary_collection(session.vector_collection_id)
        
        processing_stats = {
            'total_samples': len(session.samples),
            'processed_samples': 0,
            'failed_samples': 0,
            'total_chunks': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
        for sample in session.samples:
            try:
                start_time = datetime.now()
                self.logger.info(f"Processing sample: {sample.original_name}")
                
                # Load and process the document
                try:
                    content, metadata = self.loader.load_document(sample.file_path)
                    self.logger.info(f"Loaded document {sample.original_name}: {len(content)} chars")
                except Exception as e:
                    raise Exception(f"Failed to load document: {e}")
                
                if content and metadata:
                    # Create chunks from the content
                    try:
                        chunks = self.loader.chunk_document(content, metadata)
                        self.logger.info(f"Created {len(chunks)} chunks for {sample.original_name}")
                    except Exception as e:
                        raise Exception(f"Failed to create chunks: {e}")
                    
                    # Create a proper document object with chunks
                    document = type('Document', (), {
                        'content': content,
                        'metadata': metadata,
                        'chunks': chunks
                    })()
                else:
                    raise Exception("No content or metadata extracted from document")
                
                if document and document.chunks:
                    # Add to vector store with session collection
                    try:
                        self.logger.info(f"Adding document to collection {session.vector_collection_id}")
                        doc_id = await self.vector_store.add_document_to_collection(
                            document, 
                            collection_id=session.vector_collection_id
                        )
                        self.logger.info(f"Successfully added document with ID: {doc_id}")
                    except Exception as e:
                        raise Exception(f"Failed to add document to collection: {e}")
                    
                    # Update sample status
                    sample.processed = True
                    sample.chunks_created = len(document.chunks)
                    sample.processing_time = (datetime.now() - start_time).total_seconds()
                    
                    processing_stats['processed_samples'] += 1
                    processing_stats['total_chunks'] += len(document.chunks)
                    
                    self.logger.info(f"Processed sample {sample.original_name}: "
                                   f"{len(document.chunks)} chunks created")
                else:
                    sample.error = "Failed to extract content from document"
                    processing_stats['failed_samples'] += 1
                    processing_stats['errors'].append(f"{sample.original_name}: No content extracted")
                
            except Exception as e:
                sample.error = str(e)
                processing_stats['failed_samples'] += 1
                processing_stats['errors'].append(f"{sample.original_name}: {str(e)}")
                
                self.logger.error(f"Failed to process sample {sample.original_name}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        session.is_processed = True
        processing_stats['processing_time'] = sum(s.processing_time for s in session.samples)
        
        self.logger.info(f"Session {session_id} processing complete: "
                        f"{processing_stats['processed_samples']}/{processing_stats['total_samples']} samples processed")
        
        return processing_stats
    
    def _get_processing_stats(self, session: ContentSession) -> Dict[str, any]:
        """Get processing statistics for a session."""
        return {
            'total_samples': len(session.samples),
            'processed_samples': sum(1 for s in session.samples if s.processed),
            'failed_samples': sum(1 for s in session.samples if s.error),
            'total_chunks': sum(s.chunks_created for s in session.samples),
            'processing_time': sum(s.processing_time for s in session.samples),
            'errors': [f"{s.original_name}: {s.error}" for s in session.samples if s.error]
        }
    
    async def get_session_vector_store(self, session_id: str) -> VectorStoreManager:
        """Get a vector store manager configured for the session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if not session.is_processed:
            raise ValueError(f"Session {session_id} samples not processed yet")
        
        # Create a copy of vector store manager with session collection and correct prefix
        session_vector_store = VectorStoreManager(
            collection_name=session.vector_collection_id,
            collection_prefix=f"temp_{session.vector_collection_id}"
        )
        
        return session_vector_store
    
    def get_session_info(self, session_id: str) -> Dict[str, any]:
        """Get information about a session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        return {
            'session_id': session.session_id,
            'created_at': session.created_at.isoformat(),
            'temp_dir': str(session.temp_dir),
            'samples': [
                {
                    'original_name': s.original_name,
                    'file_size': s.file_size,
                    'processed': s.processed,
                    'chunks_created': s.chunks_created,
                    'processing_time': s.processing_time,
                    'error': s.error
                }
                for s in session.samples
            ],
            'is_processed': session.is_processed,
            'vector_collection_id': session.vector_collection_id
        }
    
    async def cleanup_session(self, session_id: str):
        """Clean up a session and its temporary files."""
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} not found for cleanup")
            return
        
        session = self.active_sessions[session_id]
        
        try:
            # Remove temporary files
            if session.temp_dir.exists():
                shutil.rmtree(session.temp_dir)
                self.logger.info(f"Removed temp directory: {session.temp_dir}")
            
            # Remove vector collection
            try:
                await self.vector_store.delete_collection(session.vector_collection_id)
                self.logger.info(f"Removed vector collection: {session.vector_collection_id}")
            except Exception as e:
                self.logger.warning(f"Failed to remove vector collection {session.vector_collection_id}: {e}")
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            self.logger.info(f"Session {session_id} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup session {session_id}: {e}")
            raise
    
    @asynccontextmanager
    async def session_context(self, session_id: str = None):
        """Context manager for handling session lifecycle."""
        if session_id is None:
            session = self.create_session()
            session_id = session.session_id
        else:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
        
        try:
            yield session
        finally:
            # Schedule cleanup (don't block)
            if session_id in self.active_sessions:
                self.active_sessions[session_id].cleanup_scheduled = True
                # Cleanup after a delay to allow for any final operations
                asyncio.create_task(self._delayed_cleanup(session_id, delay=300))  # 5 minutes delay
    
    async def _delayed_cleanup(self, session_id: str, delay: int = 300):
        """Cleanup session after delay."""
        await asyncio.sleep(delay)
        if session_id in self.active_sessions:
            await self.cleanup_session(session_id)
    
    def get_active_sessions(self) -> Dict[str, Dict[str, any]]:
        """Get information about all active sessions."""
        return {
            session_id: self.get_session_info(session_id)
            for session_id in self.active_sessions.keys()
        }
    
    async def shutdown(self):
        """Shutdown the session manager and cleanup all sessions."""
        self.logger.info("Shutting down session manager...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.cleanup_session(session_id)
        
        self.logger.info("Session manager shutdown complete")

    def debug_list_session_chunks(self, session_id: str) -> list:
        """Return a list of all chunk contents for a session's vector store."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        session = self.active_sessions[session_id]
        # Get the session-specific vector store
        session_vector_store = VectorStoreManager(
            collection_name=session.vector_collection_id,
            collection_prefix=f"temp_{session.vector_collection_id}"
        )
        # Get all chunks from the session's chunk collection
        try:
            # Directly access the ChromaVectorStore's chunks_collection
            chunk_collection = session_vector_store.vector_store.chunks_collection
            results = chunk_collection.get(limit=100)  # Adjust limit as needed
            chunks = []
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents']):
                    meta = results['metadatas'][i] if results.get('metadatas') else {}
                    chunks.append({
                        'content': doc,
                        'metadata': meta
                    })
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to list session chunks: {e}")
            return []