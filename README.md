# Erguvan AI Content Generator

A production-ready AI system that ingests competitor sustainability-related documents, learns their stylistic patterns, and generates original, client-ready content in Erguvan's brand voice.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API Key
- 16GB+ RAM (recommended for optimal performance)

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd erguvan_content_generator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. **Index competitor documents:**
```bash
python main.py index --samples-dir samples/
```

4. **Generate content:**
```bash
python main.py generate --topic "EU CBAM" --audience "CFO" --desired-length 1200
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Erguvan AI Content Generator                │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface  │  FastAPI  │  Streamlit UI  │  Configuration  │
├─────────────────────────────────────────────────────────────────┤
│                   Content Generator (RAG)                      │
├─────────────────────────────────────────────────────────────────┤
│  Document     │  Style      │  Vector       │  Quality        │
│  Loader       │  Analyzer   │  Index        │  Evaluator      │
├─────────────────────────────────────────────────────────────────┤
│  ChromaDB  │  OpenAI/GPT-4  │  Jinja2  │  Security Layer    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Ingestion**: Load competitor documents (PDF, DOCX, PPTX, MD)
2. **Style Analysis**: Extract patterns using LLM + rule-based methods
3. **Vector Indexing**: Store embeddings in ChromaDB for semantic search
4. **Content Generation**: RAG pipeline with brand voice integration
5. **Quality Evaluation**: Multi-layer assessment (plagiarism, quality, brand voice)
6. **Output**: DOCX and JSON formats with comprehensive evaluation

## 📋 Features

### Document Processing
- ✅ Multi-format support (PDF, DOCX, PPTX, Markdown)
- ✅ Security sanitization and content cleaning
- ✅ Intelligent chunking with heading-based splits
- ✅ Language detection (English/Turkish)
- ✅ Metadata extraction and document fingerprinting

### Style Analysis
- ✅ LLM-powered style pattern extraction
- ✅ Rule-based linguistic analysis
- ✅ Tone, formality, and technical level assessment
- ✅ Vocabulary and phrase pattern recognition
- ✅ Confidence scoring for analysis quality

### Vector Search
- ✅ ChromaDB integration with HNSW indexing
- ✅ <300ms search performance
- ✅ Semantic similarity search
- ✅ Context-aware chunk retrieval
- ✅ Multi-modal storage (documents, chunks, styles)

### Content Generation
- ✅ RAG pipeline with competitor knowledge
- ✅ Brand voice compliance (Erguvan guidelines)
- ✅ Multi-language support (EN/TR)
- ✅ Structured output with citations
- ✅ Audience-specific adaptation

### Quality Assurance
- ✅ Plagiarism detection (<10% threshold)
- ✅ Readability assessment (Flesch ≥50)
- ✅ Brand voice matching (≥90% threshold)
- ✅ Multi-layer evaluation pipeline
- ✅ Fail-fast quality gates

## 🔧 Usage

### Command Line Interface

```bash
# Index competitor documents
python main.py index --samples-dir samples/

# Generate content
python main.py generate \
  --topic "EU Carbon Border Adjustment Mechanism" \
  --audience "CFO" \
  --desired-length 1200 \
  --language en

# Search indexed content
python main.py search --query "carbon pricing" --limit 5

# Show system statistics
python main.py stats

# Available options
python main.py --help
```

### API Interface

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Access interactive docs
open http://localhost:8000/docs
```

### Web Interface

```bash
# Start Streamlit UI
streamlit run streamlit_app.py

# Access web interface
open http://localhost:8501
```

## 🔐 Configuration

### Models Configuration (`config/models.yaml`)

```yaml
models:
  primary:
    provider: "openai"
    generation_model: "gpt-4o"
    analysis_model: "gpt-4o-mini"
    embedding_model: "text-embedding-3-large"
    
  cost_management:
    max_cost_per_request: 0.50
    rate_limit_requests_per_minute: 60
```

### Brand Voice (`config/brand_guide.yaml`)

```yaml
voice_characteristics:
  primary_tone: "authoritative yet approachable"
  secondary_traits:
    - "solutions-oriented"
    - "technically accurate"
    - "accessible to non-experts"
    
quality_standards:
  readability:
    flesch_score: ">= 50"
  accuracy:
    fact_checking: "required for all quantitative claims"
```

## 📊 Performance Metrics

### Quality Thresholds
- **Plagiarism**: <10% similarity score
- **Readability**: ≥50 Flesch reading ease
- **Brand Voice**: ≥90% compliance score
- **Generation Time**: <60 seconds for 1200-word content

### System Performance
- **Vector Search**: <300ms response time
- **Document Processing**: ~2-5 seconds per document
- **Style Analysis**: ~10-15 seconds per document
- **Content Generation**: ~30-50 seconds per request

## 🛠️ Development

### Project Structure

```
erguvan_content_generator/
├── src/
│   ├── analyzer/          # Style analysis
│   ├── config/            # Configuration management
│   ├── evaluator/         # Quality evaluation
│   ├── generator/         # Content generation
│   ├── loader/            # Document processing
│   ├── prompts/           # Prompt templates
│   └── vector_index/      # Vector storage
├── config/                # Configuration files
├── prompts/               # Jinja2 templates
├── samples/               # Competitor documents
├── tests/                 # Test suite
├── generated/             # Output files
└── logs/                  # Application logs
```

### Adding New Document Types

1. Extend `DocumentLoader` with new parser
2. Add file type to `ALLOWED_FILE_TYPES`
3. Update security validation
4. Add tests for new format

### Extending Style Analysis

1. Add new patterns to `RuleBasedStyleAnalyzer`
2. Update LLM prompts in `prompts/system_prompts.yaml`
3. Extend `StyleProfile` dataclass
4. Update evaluation metrics

### Custom Evaluation Metrics

1. Implement new evaluator class
2. Add to `QualityEvaluator` pipeline
3. Update thresholds in configuration
4. Add monitoring dashboards

## 🔍 Monitoring

### Health Checks
```bash
curl http://localhost:8000/health
```

### Metrics (Prometheus)
```bash
curl http://localhost:8000/metrics
```

### Logging
- Application logs: `logs/erguvan_generator.log`
- API logs: `logs/api.log`
- Structured JSON format for production monitoring

## 🧪 Testing

### Run Test Suite
```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Load and stress testing
- **Security Tests**: Input validation and sanitization

## 📈 Deployment

### Production Setup

1. **Environment Configuration**:
```bash
# Production environment variables
export OPENAI_API_KEY="your-production-key"
export LOG_LEVEL="INFO"
export VECTOR_STORE_PATH="/data/vector_store"
export DATABASE_URL="postgresql://user:pass@host:5432/db"
```

2. **Docker Deployment**:
```bash
# Build container
docker build -t erguvan-generator .

# Run with production config
docker run -d \
  -p 8000:8000 \
  -v /data:/data \
  -e OPENAI_API_KEY="your-key" \
  erguvan-generator
```

3. **Scaling Considerations**:
- Use load balancer for multiple instances
- Implement Redis for caching
- Use PostgreSQL for production database
- Monitor with Prometheus + Grafana

### Security Best Practices

- ✅ API key rotation and secure storage
- ✅ Input validation and sanitization
- ✅ Rate limiting and request throttling
- ✅ Audit logging for all operations
- ✅ Content filtering and PII redaction

## 🔧 Troubleshooting

### Common Issues

**1. "Missing required environment variables"**
```bash
# Check .env file exists and contains required keys
ls -la .env
grep OPENAI_API_KEY .env
```

**2. "ChromaDB connection failed"**
```bash
# Check data directory permissions
ls -la data/
# Reset vector store if corrupted
rm -rf data/chroma_db/
```

**3. "LLM API rate limit exceeded"**
```bash
# Reduce rate limit in config/models.yaml
# Wait for rate limit reset
# Check API usage in OpenAI dashboard
```

**4. "Document processing timeout"**
```bash
# Check file size limits
# Increase timeout in configuration
# Verify file format compatibility
```

### Debug Mode
```bash
# Enable debug logging
python main.py --log-level DEBUG generate --topic "test"

# Check system health
python main.py stats
```

## 📚 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | System health check |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/upload-document` | Upload competitor document |
| `POST` | `/generate-content` | Generate content |
| `GET` | `/download/{content_id}` | Download generated content |
| `GET` | `/evaluation/{content_id}` | Get evaluation report |
| `GET` | `/search` | Search content chunks |
| `DELETE` | `/documents/{document_id}` | Delete document |

### Request/Response Examples

**Content Generation Request:**
```json
{
  "topic": "EU Carbon Border Adjustment Mechanism",
  "audience": "CFO",
  "desired_length": 1200,
  "language": "en",
  "style_override": "executive summary format"
}
```

**Content Generation Response:**
```json
{
  "content_id": "content_1234567890",
  "title": "EU CBAM: Strategic Implications for CFOs",
  "word_count": 1247,
  "generation_time_seconds": 45.2,
  "evaluation_summary": {
    "overall_recommendation": "pass",
    "passes_all_thresholds": true,
    "plagiarism_score": 3.2,
    "quality_score": 87.5,
    "brand_voice_score": 92.1
  },
  "download_url": "/download/content_1234567890"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include type hints
- Write tests for new features
- Update documentation

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Technical Issues**: Create GitHub issue
- **Feature Requests**: Submit enhancement proposal
- **Security Issues**: Email security@erguvan.com

---

**Built with ❤️ by the Erguvan Team**

*Empowering businesses to navigate climate regulations and achieve sustainability goals through AI-powered content generation.*