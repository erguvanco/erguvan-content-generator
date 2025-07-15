#!/usr/bin/env python3
"""
Demonstration of Erguvan AI Content Generator Components
Shows document loading, configuration, and prompt template system
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config_manager
from src.loader.document_loader import DocumentLoader, DocumentBatch
from src.prompts import prompt_manager

def main():
    print("🚀 Erguvan AI Content Generator - Component Demonstration")
    print("=" * 60)
    
    # 1. Configuration System Demo
    print("\n1. 📋 Configuration System")
    print("-" * 30)
    
    try:
        # Load model configuration
        model_config = config_manager.get_model_config()
        print(f"✅ Model Config Loaded:")
        print(f"   - Provider: {model_config.provider}")
        print(f"   - Generation Model: {model_config.generation_model}")
        print(f"   - Analysis Model: {model_config.analysis_model}")
        print(f"   - Max Cost per Request: ${model_config.max_cost_per_request}")
        
        # Load brand configuration
        brand_config = config_manager.get_brand_config()
        print(f"\n✅ Brand Config Loaded:")
        print(f"   - Primary Tone: {brand_config.primary_tone}")
        print(f"   - Secondary Traits: {', '.join(brand_config.secondary_traits[:3])}...")
        print(f"   - Flesch Score Min: {brand_config.flesch_score_min}")
        
        # Validate environment
        config_manager.validate_environment()
        print("✅ Environment validation passed")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # 2. Document Loader Demo
    print("\n2. 📄 Document Loader System")
    print("-" * 30)
    
    samples_dir = Path("samples")
    if not samples_dir.exists():
        print("❌ Samples directory not found")
        return
    
    loader = DocumentLoader()
    batch_processor = DocumentBatch(loader)
    
    # Load all documents from samples directory
    try:
        documents = batch_processor.load_directory(str(samples_dir))
        
        print(f"✅ Loaded {len(documents)} documents:")
        
        for i, (content, metadata) in enumerate(documents, 1):
            print(f"\n   Document {i}: {metadata.file_name}")
            print(f"   - Type: {metadata.file_type}")
            print(f"   - Size: {metadata.file_size:,} bytes")
            print(f"   - Language: {metadata.language}")
            print(f"   - Word Count: {metadata.word_count:,}")
            print(f"   - Document ID: {metadata.document_id[:8]}...")
            
            # Show first 150 characters of content
            preview = content[:150].replace('\n', ' ')
            print(f"   - Content Preview: {preview}...")
            
            # Demonstrate chunking
            chunks = loader.chunk_document(content, metadata, chunk_size=200, overlap=50)
            print(f"   - Chunks Created: {len(chunks)}")
            
            if chunks:
                print(f"   - First Chunk: {chunks[0].content[:100]}...")
                if chunks[0].heading:
                    print(f"   - Heading: {chunks[0].heading}")
        
    except Exception as e:
        print(f"❌ Document loading error: {e}")
        return
    
    # 3. Prompt Templates Demo
    print("\n3. 🎯 Prompt Templates System")
    print("-" * 30)
    
    try:
        # Show available templates
        template_stats = prompt_manager.get_template_stats()
        print(f"✅ Available Templates: {len(template_stats)}")
        
        for name, stats in template_stats.items():
            print(f"   - {name}: {stats['description']}")
        
        # Demonstrate style analysis prompt
        if documents:
            sample_content = documents[0][0][:1000]  # First 1000 chars
            
            print(f"\n📝 Style Analysis Prompt Example:")
            style_prompt = prompt_manager.render_style_analysis_prompt(sample_content)
            
            # Show first 300 characters of rendered prompt
            prompt_preview = style_prompt[:300].replace('\n', ' ')
            print(f"   Rendered Prompt: {prompt_preview}...")
        
        # Demonstrate content generation prompt
        print(f"\n📝 Content Generation Prompt Example:")
        
        # Create mock context chunks
        mock_chunks = []
        if documents:
            for i, (content, metadata) in enumerate(documents[:2]):  # Use first 2 docs
                chunk = type('MockChunk', (), {
                    'content': content[:500],
                    'metadata': metadata,
                    'relevance_score': 0.85 - (i * 0.1)
                })()
                mock_chunks.append(chunk)
        
        content_prompt = prompt_manager.render_content_generation_prompt(
            topic="EU Carbon Border Adjustment Mechanism",
            audience="CFO",
            desired_length=1200,
            context_chunks=mock_chunks,
            style_patterns=["authoritative tone", "executive summary format", "quantified benefits"],
            language="en"
        )
        
        prompt_preview = content_prompt[:400].replace('\n', ' ')
        print(f"   Rendered Prompt: {prompt_preview}...")
        
        # Show brand voice evaluation prompt
        print(f"\n📝 Brand Voice Evaluation Prompt Example:")
        eval_prompt = prompt_manager.render_brand_voice_evaluation_prompt(
            "This is sample content for brand voice evaluation..."
        )
        
        prompt_preview = eval_prompt[:300].replace('\n', ' ')
        print(f"   Rendered Prompt: {prompt_preview}...")
        
    except Exception as e:
        print(f"❌ Prompt templates error: {e}")
        return
    
    # 4. Security and Sanitization Demo
    print("\n4. 🔒 Security Features")
    print("-" * 30)
    
    try:
        from src.loader.document_loader import DocumentSanitizer
        
        # Test content sanitization
        test_content = """
        This document contains sensitive information like john.doe@example.com
        and phone number 555-123-4567. Credit card: 1234-5678-9012-3456.
        
        © 2024 Company Name. All rights reserved.
        Confidential - Not for distribution.
        
        <script>alert('xss')</script>
        <p>Some HTML content</p>
        """
        
        sanitized = DocumentSanitizer.sanitize_content(test_content)
        print("✅ Content Sanitization:")
        print(f"   - Original length: {len(test_content)} chars")
        print(f"   - Sanitized length: {len(sanitized)} chars")
        print(f"   - Sanitized content: {sanitized[:200]}...")
        
        # Test file validation
        for content, metadata in documents:
            file_path = Path(metadata.file_path)
            is_valid = DocumentSanitizer.validate_file_security(file_path)
            print(f"   - {metadata.file_name}: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
    except Exception as e:
        print(f"❌ Security demo error: {e}")
    
    # 5. Summary and Next Steps
    print("\n5. 📊 Component Summary")
    print("-" * 30)
    
    print("✅ Completed Components:")
    print("   • Configuration Management (models.yaml, brand_guide.yaml)")
    print("   • Document Loader (PDF, DOCX, PPTX, MD support)")
    print("   • Security Sanitization (content cleaning, file validation)")
    print("   • Prompt Templates (Jinja2, centralized, brand-aware)")
    print("   • Environment Management (API keys, settings)")
    
    print("\n🔄 Next Implementation Steps:")
    print("   • Style Analyzer (LLM + rule-based)")
    print("   • Vector Index System (ChromaDB)")
    print("   • Content Generator (RAG pipeline)")
    print("   • Quality Evaluator (multi-layer)")
    print("   • API & CLI Interface")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"   - Documents processed: {len(documents) if 'documents' in locals() else 0}")
    print(f"   - Templates available: {len(template_stats) if 'template_stats' in locals() else 0}")
    print(f"   - System ready for next phase")

if __name__ == "__main__":
    main()