#!/usr/bin/env python3
"""
Erguvan AI Content Generator - Main CLI Entry Point
Production-ready command-line interface with comprehensive options.
"""

import argparse
import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config_manager
from src.loader.markitdown_loader import MarkItDownLoader, DocumentBatch
from src.analyzer.style_analyzer import StyleAnalyzer, StyleProfileManager
from src.vector_index.vector_store import VectorStoreManager
from src.generator.content_generator import ContentGenerator, ContentRequest
from src.evaluator.quality_evaluator import QualityEvaluator


def setup_logging(log_level: str = "INFO"):
    """Configure structured logging."""
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/erguvan_generator.log'),
            logging.StreamHandler()
        ]
    )


async def index_documents(args):
    """Index documents from samples directory."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        vector_store_manager = VectorStoreManager()
        style_manager = StyleProfileManager()
        
        # Load documents
        loader = MarkItDownLoader()
        batch = DocumentBatch(loader)
        
        samples_dir = Path(args.samples_dir)
        if not samples_dir.exists():
            logger.error(f"Samples directory not found: {samples_dir}")
            return False
        
        logger.info(f"Loading documents from: {samples_dir}")
        documents = batch.load_directory(str(samples_dir))
        
        if not documents:
            logger.warning("No documents found to index")
            return False
        
        # Prepare chunks
        chunks_list = []
        for content, metadata in documents:
            chunks = loader.chunk_document(content, metadata)
            chunks_list.append(chunks)
        
        # Index documents
        logger.info("Indexing documents in vector store...")
        indexed_docs = await vector_store_manager.index_documents(documents, chunks_list)
        
        # Analyze styles
        logger.info("Analyzing document styles...")
        style_profiles = await style_manager.analyze_documents(documents)
        
        # Index style profiles
        indexed_styles = await vector_store_manager.index_style_profiles(style_profiles)
        
        # Save style profiles
        style_manager.save_profiles("data/style_profiles.json")
        
        logger.info(f"Indexing completed successfully:")
        logger.info(f"  - Documents indexed: {len(indexed_docs)}")
        logger.info(f"  - Style profiles created: {len(indexed_styles)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return False


async def generate_content(args):
    """Generate content based on arguments."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        vector_store_manager = VectorStoreManager()
        style_manager = StyleProfileManager()
        
        # Load existing style profiles if available
        style_profiles_path = Path("data/style_profiles.json")
        if style_profiles_path.exists():
            style_manager.load_profiles(str(style_profiles_path))
        
        # Initialize generator and evaluator
        generator = ContentGenerator(vector_store_manager, style_manager)
        evaluator = QualityEvaluator()
        
        # Create content request
        content_request = ContentRequest(
            topic=args.topic,
            audience=args.audience,
            desired_length=args.desired_length,
            language=args.language,
            style_override=args.style_override
        )
        
        # Generate content
        logger.info(f"Generating content for topic: {args.topic}")
        start_time = datetime.now()
        
        generated_content = await generator.generate_content(content_request)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Content generation completed in {generation_time:.2f}s")
        
        # Get context chunks for evaluation
        context_chunks = await vector_store_manager.search_for_generation(
            topic=args.topic,
            audience=args.audience,
            language=args.language,
            max_chunks=5
        )
        
        # Evaluate content
        logger.info("Evaluating content quality...")
        evaluation = await evaluator.evaluate_content(generated_content, context_chunks)
        
        # Check if content passes all thresholds
        if not evaluation.passes_all_thresholds and not args.force:
            logger.error("Content failed quality evaluation")
            logger.error(f"Recommendation: {evaluation.overall_recommendation}")
            logger.error(f"Plagiarism score: {evaluation.plagiarism_report.overall_score:.1f}%")
            logger.error(f"Quality score: {evaluation.quality_report.overall_score:.1f}%")
            logger.error(f"Brand voice score: {evaluation.brand_voice_report.overall_score:.1f}%")
            
            if not args.save_failed:
                sys.exit(1)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = args.topic.replace(' ', '_').lower()
        filename_base = f"{safe_topic}_{args.audience.lower()}_{timestamp}"
        
        # Save content to DOCX
        docx_path = output_dir / f"{filename_base}.docx"
        await generator.save_to_docx(generated_content, docx_path)
        
        # Save content to JSON
        json_path = output_dir / f"{filename_base}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(generated_content.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save evaluation report
        eval_path = output_dir / f"{filename_base}_evaluation.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n✅ Content Generation Summary")
        print(f"Topic: {args.topic}")
        print(f"Audience: {args.audience}")
        print(f"Word Count: {generated_content.word_count:,}")
        print(f"Generation Time: {generation_time:.2f}s")
        print(f"")
        print(f"📊 Quality Evaluation:")
        print(f"  Overall Recommendation: {evaluation.overall_recommendation.upper()}")
        print(f"  Plagiarism Score: {evaluation.plagiarism_report.overall_score:.1f}%")
        print(f"  Quality Score: {evaluation.quality_report.overall_score:.1f}%")
        print(f"  Brand Voice Score: {evaluation.brand_voice_report.overall_score:.1f}%")
        print(f"")
        print(f"📁 Output Files:")
        print(f"  Content: {docx_path}")
        print(f"  JSON: {json_path}")
        print(f"  Evaluation: {eval_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return False


async def search_content(args):
    """Search for content in the vector store."""
    logger = logging.getLogger(__name__)
    
    try:
        vector_store_manager = VectorStoreManager()
        
        filters = {}
        if args.language:
            filters["language"] = args.language
        
        results = await vector_store_manager.vector_store.search_chunks(
            query=args.query,
            limit=args.limit,
            filters=filters
        )
        
        print(f"\n🔍 Search Results for: '{args.query}'")
        print(f"Found {len(results)} results")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Relevance: {result.relevance_score:.3f}")
            print(f"   Document: {result.metadata.get('file_name', 'Unknown')}")
            if result.heading:
                print(f"   Heading: {result.heading}")
            print(f"   Content: {result.content[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return False


async def show_stats(args):
    """Show system statistics."""
    logger = logging.getLogger(__name__)
    
    try:
        vector_store_manager = VectorStoreManager()
        stats = vector_store_manager.get_stats()
        
        print(f"\n📊 System Statistics")
        print(f"Total Documents: {stats.total_documents:,}")
        print(f"Total Chunks: {stats.total_chunks:,}")
        print(f"Total Embeddings: {stats.total_embeddings:,}")
        print(f"Index Size: {stats.index_size_mb:.2f} MB")
        print(f"Languages: {', '.join(stats.languages) if stats.languages else 'None'}")
        print(f"Average Chunk Size: {stats.average_chunk_size:.1f} characters")
        print(f"Last Updated: {stats.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Erguvan AI Content Generator - Production CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python main.py index --samples-dir samples/
  
  # Generate content
  python main.py generate --topic "EU CBAM" --audience "CFO" --desired-length 1200
  
  # Search content
  python main.py search --query "carbon pricing" --limit 5
  
  # Show statistics
  python main.py stats
        """
    )
    
    # Global arguments
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--config-dir', help='Configuration directory')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('--samples-dir', default='samples',
                             help='Directory containing competitor documents')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate content')
    generate_parser.add_argument('--topic', required=True, 
                                help='Content topic (e.g., "EU CBAM")')
    generate_parser.add_argument('--audience', required=True, 
                                help='Target audience (e.g., "CFO")')
    generate_parser.add_argument('--desired-length', type=int, required=True, 
                                help='Desired word count')
    generate_parser.add_argument('--style-override', 
                                help='Optional style override')
    generate_parser.add_argument('--language', default='en', 
                                help='Output language (en/tr)')
    generate_parser.add_argument('--output-dir', default='generated', 
                                help='Output directory')
    generate_parser.add_argument('--force', action='store_true',
                                help='Generate even if quality thresholds fail')
    generate_parser.add_argument('--save-failed', action='store_true',
                                help='Save content even if evaluation fails')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search content')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Result limit')
    search_parser.add_argument('--language', help='Filter by language')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run appropriate command
    if args.command == 'index':
        success = asyncio.run(index_documents(args))
    elif args.command == 'generate':
        success = asyncio.run(generate_content(args))
    elif args.command == 'search':
        success = asyncio.run(search_content(args))
    elif args.command == 'stats':
        success = asyncio.run(show_stats(args))
    else:
        parser.print_help()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()