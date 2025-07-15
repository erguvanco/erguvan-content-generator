#!/usr/bin/env python3
"""
Simple test of Erguvan AI Content Generator components
Test environment and component loading without API calls
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_environment():
    """Test environment variable loading."""
    print("🔧 Environment Test")
    print("-" * 20)
    
    # Check if API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OPENAI_API_KEY loaded: {api_key[:20]}...{api_key[-10:]}")
    else:
        print("❌ OPENAI_API_KEY not found in environment")
    
    # Check other environment variables
    other_vars = ['LOG_LEVEL', 'MAX_FILE_SIZE_MB', 'ALLOWED_FILE_TYPES']
    for var in other_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: not set")

def test_configuration():
    """Test configuration system without API validation."""
    print("\n📋 Configuration Test")
    print("-" * 20)
    
    try:
        from src.config import config_manager
        
        # Load configurations
        models_config = config_manager.load_models_config()
        brand_config = config_manager.load_brand_config()
        
        print("✅ Models config loaded successfully")
        print(f"   - Generation model: {models_config['models']['primary']['generation_model']}")
        
        print("✅ Brand config loaded successfully") 
        print(f"   - Primary tone: {brand_config['voice_characteristics']['primary_tone']}")
        
        # Test config objects
        model_config = config_manager.get_model_config()
        brand_config_obj = config_manager.get_brand_config()
        
        print("✅ Configuration objects created successfully")
        print(f"   - Model: {model_config.generation_model}")
        print(f"   - Brand tone: {brand_config_obj.primary_tone}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")

def test_document_loading():
    """Test document loading without processing."""
    print("\n📄 Document Loading Test")
    print("-" * 20)
    
    try:
        from src.loader.document_loader import DocumentLoader, DocumentBatch
        
        loader = DocumentLoader()
        batch = DocumentBatch(loader)
        
        print("✅ Document loader created successfully")
        
        # Check samples directory
        samples_dir = Path("samples")
        if samples_dir.exists():
            files = list(samples_dir.glob("*.docx"))
            print(f"✅ Found {len(files)} DOCX files in samples/")
            
            # Try to load one document
            if files:
                try:
                    content, metadata = loader.load_document(str(files[0]))
                    print(f"✅ Successfully loaded: {metadata.file_name}")
                    print(f"   - Size: {len(content)} characters")
                    print(f"   - Language: {metadata.language}")
                    print(f"   - Word count: {metadata.word_count}")
                    
                    # Test chunking
                    chunks = loader.chunk_document(content, metadata, chunk_size=200)
                    print(f"✅ Created {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"❌ Document loading error: {e}")
        else:
            print("❌ Samples directory not found")
            
    except Exception as e:
        print(f"❌ Document loader import error: {e}")

def test_prompt_templates():
    """Test prompt template system."""
    print("\n🎯 Prompt Templates Test")
    print("-" * 20)
    
    try:
        from src.prompts import prompt_manager
        
        # Check if templates are loaded
        stats = prompt_manager.get_template_stats()
        print(f"✅ Loaded {len(stats)} prompt templates:")
        
        for name in stats.keys():
            print(f"   - {name}")
        
        # Test template rendering
        if 'base_system_prompt' in stats:
            try:
                prompt = prompt_manager.render_system_prompt('base_system_prompt')
                print(f"✅ Template rendering successful")
                print(f"   - Rendered length: {len(prompt)} characters")
            except Exception as e:
                print(f"❌ Template rendering error: {e}")
        
    except Exception as e:
        print(f"❌ Prompt templates error: {e}")

def test_security():
    """Test security features."""
    print("\n🔒 Security Test")
    print("-" * 20)
    
    try:
        from src.loader.document_loader import DocumentSanitizer
        
        # Test content sanitization
        test_content = """
        Test document with email@example.com and phone 555-123-4567.
        <script>alert('test')</script>
        <p>HTML content</p>
        © 2024 Test Company
        """
        
        sanitized = DocumentSanitizer.sanitize_content(test_content)
        print("✅ Content sanitization working")
        print(f"   - Original: {len(test_content)} chars")
        print(f"   - Sanitized: {len(sanitized)} chars")
        print(f"   - Sanitized preview: {sanitized[:100]}...")
        
    except Exception as e:
        print(f"❌ Security test error: {e}")

def main():
    """Run all component tests."""
    print("🚀 Erguvan AI Content Generator - Component Test")
    print("=" * 50)
    
    test_environment()
    test_configuration()
    test_document_loading()
    test_prompt_templates()
    test_security()
    
    print("\n✅ Component testing completed!")

if __name__ == "__main__":
    main()