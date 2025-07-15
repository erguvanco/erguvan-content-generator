"""
FastAPI application for Erguvan AI Content Generator.
Production-ready API with comprehensive endpoints and middleware.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config_manager
from src.loader.document_loader import DocumentLoader, DocumentBatch
from src.analyzer.style_analyzer import StyleAnalyzer, StyleProfileManager
from src.vector_index.vector_store import VectorStoreManager
from src.generator.content_generator import ContentGenerator, ContentRequest
from src.evaluator.quality_evaluator import QualityEvaluator


# Prometheus metrics
REQUEST_COUNT = Counter('erguvan_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('erguvan_request_duration_seconds', 'Request duration')
GENERATION_COUNT = Counter('erguvan_generations_total', 'Total content generations', ['status'])
GENERATION_DURATION = Histogram('erguvan_generation_duration_seconds', 'Content generation duration')


# Pydantic models for API
class ContentGenerationRequest(BaseModel):
    """Request model for content generation."""
    topic: str = Field(..., description="Content topic", example="EU Carbon Border Adjustment Mechanism")
    audience: str = Field(..., description="Target audience", example="CFO")
    desired_length: int = Field(..., description="Desired word count", example=1200, ge=100, le=10000)
    language: str = Field("en", description="Content language", example="en")
    style_override: Optional[str] = Field(None, description="Optional style override")
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "EU Carbon Border Adjustment Mechanism",
                "audience": "CFO",
                "desired_length": 1200,
                "language": "en",
                "style_override": "executive summary format"
            }
        }


class ContentGenerationResponse(BaseModel):
    """Response model for content generation."""
    content_id: str
    title: str
    word_count: int
    sections: List[Dict[str, Any]]
    key_takeaways: List[str]
    generation_time_seconds: float
    evaluation_summary: Dict[str, Any]
    download_url: str


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    word_count: int
    chunks_created: int
    language: str
    processing_time_seconds: float


class SystemHealthResponse(BaseModel):
    """Response model for system health check."""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, Any]
    vector_store_stats: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    timestamp: datetime


# FastAPI app initialization
app = FastAPI(
    title="Erguvan AI Content Generator",
    description="Production-ready AI content generation system for sustainability advisory services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global components
vector_store_manager = None
style_manager = None
content_generator = None
quality_evaluator = None

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    global vector_store_manager, style_manager, content_generator, quality_evaluator
    
    logger.info("Starting Erguvan AI Content Generator API")
    
    try:
        # Initialize components
        vector_store_manager = VectorStoreManager()
        style_manager = StyleProfileManager()
        content_generator = ContentGenerator(vector_store_manager, style_manager)
        quality_evaluator = QualityEvaluator()
        
        # Create required directories
        os.makedirs("generated", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware to collect metrics."""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    # Record metrics
    duration = (datetime.now() - start_time).total_seconds()
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Erguvan AI Content Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """System health check endpoint."""
    try:
        # Check vector store health
        vector_store_health = vector_store_manager.health_check()
        vector_store_stats = vector_store_manager.get_stats()
        
        # Check configuration
        config_healthy = True
        try:
            config_manager.validate_environment()
        except Exception:
            config_healthy = False
        
        components = {
            "vector_store": vector_store_health["healthy"],
            "configuration": config_healthy,
            "content_generator": content_generator is not None,
            "quality_evaluator": quality_evaluator is not None
        }
        
        overall_status = "healthy" if all(components.values()) else "degraded"
        
        return SystemHealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="1.0.0",
            components=components,
            vector_store_stats=vector_store_stats.__dict__
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload")
):
    """Upload and process a competitor document."""
    start_time = datetime.now()
    
    try:
        # Validate file type
        allowed_types = ['.pdf', '.docx', '.pptx', '.md']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Save uploaded file
        upload_path = Path("samples") / file.filename
        upload_path.parent.mkdir(exist_ok=True)
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        loader = DocumentLoader()
        content_text, metadata = loader.load_document(str(upload_path))
        chunks = loader.chunk_document(content_text, metadata)
        
        # Index document (background task)
        background_tasks.add_task(
            index_document_background,
            content_text, metadata, chunks
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DocumentUploadResponse(
            document_id=metadata.document_id,
            filename=file.filename,
            word_count=metadata.word_count,
            chunks_created=len(chunks),
            language=metadata.language,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def index_document_background(content: str, metadata, chunks: List):
    """Background task to index document."""
    try:
        # Add to vector store
        await vector_store_manager.vector_store.add_document(content, metadata, chunks)
        
        # Analyze style
        style_analyzer = StyleAnalyzer()
        style_profile = await style_analyzer.analyze_document(content, metadata)
        
        # Add style profile to manager and vector store
        style_manager.profiles[metadata.document_id] = style_profile
        await vector_store_manager.vector_store.add_style_profile(style_profile)
        
        logger.info(f"Document indexed successfully: {metadata.file_name}")
        
    except Exception as e:
        logger.error(f"Background indexing failed: {e}")


@app.post("/generate-content", response_model=ContentGenerationResponse)
async def generate_content(request: ContentGenerationRequest):
    """Generate content based on request."""
    start_time = datetime.now()
    
    try:
        # Convert to internal request format
        content_request = ContentRequest(
            topic=request.topic,
            audience=request.audience,
            desired_length=request.desired_length,
            language=request.language,
            style_override=request.style_override
        )
        
        # Generate content
        GENERATION_COUNT.labels(status="started").inc()
        
        generated_content = await content_generator.generate_content(content_request)
        
        # Get context chunks for evaluation
        context_chunks = await vector_store_manager.search_for_generation(
            topic=request.topic,
            audience=request.audience,
            language=request.language,
            max_chunks=5
        )
        
        # Evaluate content
        evaluation = await quality_evaluator.evaluate_content(generated_content, context_chunks)
        
        # Save to file
        content_id = f"content_{int(start_time.timestamp())}"
        output_path = Path("generated") / f"{content_id}.docx"
        await content_generator.save_to_docx(generated_content, output_path)
        
        # Save evaluation report
        eval_path = Path("generated") / f"{content_id}_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation.to_dict(), f, indent=2)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        GENERATION_DURATION.observe(generation_time)
        GENERATION_COUNT.labels(status="completed").inc()
        
        return ContentGenerationResponse(
            content_id=content_id,
            title=generated_content.title,
            word_count=generated_content.word_count,
            sections=[{
                "heading": section.heading,
                "word_count": section.word_count
            } for section in generated_content.sections],
            key_takeaways=generated_content.key_takeaways,
            generation_time_seconds=generation_time,
            evaluation_summary={
                "overall_recommendation": evaluation.overall_recommendation,
                "passes_all_thresholds": evaluation.passes_all_thresholds,
                "plagiarism_score": evaluation.plagiarism_report.overall_score,
                "quality_score": evaluation.quality_report.overall_score,
                "brand_voice_score": evaluation.brand_voice_report.overall_score
            },
            download_url=f"/download/{content_id}"
        )
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        GENERATION_COUNT.labels(status="failed").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{content_id}")
async def download_content(content_id: str):
    """Download generated content file."""
    try:
        file_path = Path("generated") / f"{content_id}.docx"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Content not found")
        
        return FileResponse(
            path=file_path,
            filename=f"{content_id}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/{content_id}")
async def get_evaluation(content_id: str):
    """Get evaluation report for generated content."""
    try:
        eval_path = Path("generated") / f"{content_id}_evaluation.json"
        
        if not eval_path.exists():
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        with open(eval_path, 'r') as f:
            evaluation_data = json.load(f)
        
        return JSONResponse(content=evaluation_data)
        
    except Exception as e:
        logger.error(f"Evaluation retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    try:
        stats = vector_store_manager.get_stats()
        
        return {
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "languages": stats.languages,
            "last_updated": stats.last_updated.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_content(
    query: str,
    limit: int = 10,
    language: Optional[str] = None
):
    """Search for content chunks."""
    try:
        filters = {}
        if language:
            filters["language"] = language
        
        results = await vector_store_manager.vector_store.search_chunks(
            query=query,
            limit=limit,
            filters=filters
        )
        
        return {
            "query": query,
            "results": [{
                "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "relevance_score": result.relevance_score,
                "document_id": result.document_id,
                "heading": result.heading,
                "metadata": result.metadata
            } for result in results]
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the system."""
    try:
        success = vector_store_manager.vector_store.delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )