"""
Production-grade style analyzer with LLM + rule-based evaluation.
Extracts stylistic patterns from competitor documents for content generation.
"""

import re
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

import textstat
from openai import OpenAI
from anthropic import Anthropic

from src.config import config_manager
from src.prompts import prompt_manager
from src.loader.document_loader import DocumentMetadata


@dataclass
class StyleProfile:
    """Comprehensive style profile for a document."""
    document_id: str
    file_name: str
    language: str
    
    # LLM-analyzed features
    tone: str
    formality_score: float
    technical_level: str
    common_phrases: List[str]
    sentence_starters: List[str]
    transition_words: List[str]
    citation_style: str
    
    # Rule-based features
    avg_sentence_length: float
    avg_paragraph_length: float
    readability_score: float
    passive_voice_ratio: float
    question_ratio: float
    exclamation_ratio: float
    
    # Structure analysis
    heading_patterns: List[str]
    list_usage: float
    bullet_point_usage: float
    
    # Vocabulary analysis
    technical_terms: List[str]
    action_verbs: List[str]
    sentiment_words: List[str]
    
    # Metadata
    analyzed_at: datetime
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'document_id': self.document_id,
            'file_name': self.file_name,
            'language': self.language,
            'tone': self.tone,
            'formality_score': self.formality_score,
            'technical_level': self.technical_level,
            'common_phrases': self.common_phrases,
            'sentence_starters': self.sentence_starters,
            'transition_words': self.transition_words,
            'citation_style': self.citation_style,
            'avg_sentence_length': self.avg_sentence_length,
            'avg_paragraph_length': self.avg_paragraph_length,
            'readability_score': self.readability_score,
            'passive_voice_ratio': self.passive_voice_ratio,
            'question_ratio': self.question_ratio,
            'exclamation_ratio': self.exclamation_ratio,
            'heading_patterns': self.heading_patterns,
            'list_usage': self.list_usage,
            'bullet_point_usage': self.bullet_point_usage,
            'technical_terms': self.technical_terms,
            'action_verbs': self.action_verbs,
            'sentiment_words': self.sentiment_words,
            'analyzed_at': self.analyzed_at.isoformat(),
            'confidence_score': self.confidence_score
        }


class RuleBasedStyleAnalyzer:
    """Rule-based style analysis for objective metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common patterns for analysis
        self.passive_voice_patterns = [
            r'\b(is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(is|are|was|were|been|being)\s+\w+en\b',
        ]
        
        self.technical_sustainability_terms = [
            'carbon footprint', 'greenhouse gas', 'scope 1', 'scope 2', 'scope 3',
            'net zero', 'carbon neutral', 'decarbonization', 'emissions trading',
            'carbon border adjustment', 'cbam', 'eu ets', 'tcfd', 'csrd',
            'science based targets', 'sbti', 'lifecycle assessment', 'lca',
            'environmental impact', 'sustainability reporting', 'esg'
        ]
        
        self.action_verbs = [
            'implement', 'achieve', 'reduce', 'measure', 'report', 'assess',
            'monitor', 'track', 'optimize', 'improve', 'develop', 'establish',
            'calculate', 'quantify', 'verify', 'validate', 'analyze', 'evaluate'
        ]
        
        self.transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'meanwhile', 'nevertheless', 'accordingly', 'hence',
            'thus', 'subsequently', 'alternatively', 'specifically', 'notably'
        ]
    
    def analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure and formatting."""
        lines = content.split('\n')
        sentences = self._split_sentences(content)
        paragraphs = self._split_paragraphs(content)
        
        # Basic metrics
        total_sentences = len(sentences)
        total_words = len(content.split())
        
        # Sentence analysis
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        avg_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 0
        
        # Paragraph analysis
        paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
        avg_paragraph_length = statistics.mean(paragraph_lengths) if paragraph_lengths else 0
        
        # Readability
        readability_score = textstat.flesch_reading_ease(content)
        
        # Voice analysis
        passive_voice_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                                for pattern in self.passive_voice_patterns)
        passive_voice_ratio = passive_voice_count / total_sentences if total_sentences > 0 else 0
        
        # Punctuation analysis
        question_count = content.count('?')
        exclamation_count = content.count('!')
        question_ratio = question_count / total_sentences if total_sentences > 0 else 0
        exclamation_ratio = exclamation_count / total_sentences if total_sentences > 0 else 0
        
        # List and bullet point usage
        bullet_lines = [line for line in lines if re.match(r'^\s*[•\-\*\+]\s+', line)]
        numbered_lines = [line for line in lines if re.match(r'^\s*\d+\.\s+', line)]
        list_usage = (len(bullet_lines) + len(numbered_lines)) / len(lines) if lines else 0
        bullet_point_usage = len(bullet_lines) / len(lines) if lines else 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_paragraph_length': avg_paragraph_length,
            'readability_score': readability_score,
            'passive_voice_ratio': passive_voice_ratio,
            'question_ratio': question_ratio,
            'exclamation_ratio': exclamation_ratio,
            'list_usage': list_usage,
            'bullet_point_usage': bullet_point_usage,
            'total_sentences': total_sentences,
            'total_words': total_words
        }
    
    def analyze_vocabulary(self, content: str) -> Dict[str, List[str]]:
        """Analyze vocabulary patterns and word usage."""
        words = content.lower().split()
        
        # Find technical terms
        technical_terms = []
        for term in self.technical_sustainability_terms:
            if term in content.lower():
                technical_terms.append(term)
        
        # Find action verbs
        found_action_verbs = []
        for verb in self.action_verbs:
            if verb in words:
                found_action_verbs.append(verb)
        
        # Find transition words
        found_transitions = []
        for word in self.transition_words:
            if word in words:
                found_transitions.append(word)
        
        # Extract sentence starters
        sentences = self._split_sentences(content)
        sentence_starters = []
        for sentence in sentences[:10]:  # First 10 sentences
            words = sentence.strip().split()
            if words:
                starter = words[0].lower()
                if len(starter) > 2:  # Ignore very short words
                    sentence_starters.append(starter)
        
        return {
            'technical_terms': technical_terms,
            'action_verbs': found_action_verbs,
            'transition_words': found_transitions,
            'sentence_starters': list(set(sentence_starters))
        }
    
    def extract_headings(self, content: str) -> List[str]:
        """Extract heading patterns from content."""
        lines = content.split('\n')
        headings = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for various heading patterns
            if re.match(r'^#{1,6}\s+', line):  # Markdown headings
                headings.append(line)
            elif re.match(r'^[A-Z][A-Z\s]+$', line) and len(line) < 100:  # ALL CAPS
                headings.append(line)
            elif re.match(r'^\d+\.\s+[A-Z]', line):  # Numbered headings
                headings.append(line)
            elif re.match(r'^[A-Z][a-z]+.*:$', line):  # Colon-ending headings
                headings.append(line)
        
        return headings
    
    def _split_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]


class LLMStyleAnalyzer:
    """LLM-powered style analysis for subjective features."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config_manager.get_model_config()
        
        # Initialize LLM clients
        if self.config.provider == "openai":
            self.client = OpenAI()
        elif self.config.provider == "anthropic":
            self.client = Anthropic()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def analyze_style(self, content: str) -> Dict[str, Any]:
        """Analyze document style using LLM."""
        try:
            # Generate style analysis prompt
            prompt = prompt_manager.render_style_analysis_prompt(content)
            
            # Call LLM
            response = await self._call_llm(prompt)
            
            # Parse JSON response
            style_data = json.loads(response)
            
            # Validate response structure
            required_fields = ['tone', 'formality_score', 'technical_level', 
                             'common_phrases', 'sentence_starters', 'citation_style']
            
            for field in required_fields:
                if field not in style_data:
                    self.logger.warning(f"Missing field in LLM response: {field}")
                    style_data[field] = "unknown" if field != 'formality_score' else 5.0
            
            return style_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return self._get_fallback_style()
        except Exception as e:
            self.logger.error(f"LLM style analysis failed: {e}")
            return self._get_fallback_style()
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API with error handling and retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.config.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.config.analysis_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
                
                elif self.config.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.config.analysis_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    return response.content[0].text
                
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _get_fallback_style(self) -> Dict[str, Any]:
        """Return fallback style data when LLM fails."""
        return {
            'tone': 'professional',
            'formality_score': 7.0,
            'technical_level': 'intermediate',
            'common_phrases': ['carbon emissions', 'climate change', 'sustainability'],
            'sentence_starters': ['the', 'this', 'in', 'for'],
            'transition_words': ['however', 'therefore', 'furthermore'],
            'citation_style': 'academic'
        }


class StyleAnalyzer:
    """Main style analyzer combining LLM and rule-based analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rule_analyzer = RuleBasedStyleAnalyzer()
        self.llm_analyzer = LLMStyleAnalyzer()
    
    async def analyze_document(self, content: str, metadata: DocumentMetadata) -> StyleProfile:
        """Perform comprehensive style analysis on a document."""
        self.logger.info(f"Analyzing style for document: {metadata.file_name}")
        
        # Rule-based analysis
        structure_analysis = self.rule_analyzer.analyze_structure(content)
        vocabulary_analysis = self.rule_analyzer.analyze_vocabulary(content)
        headings = self.rule_analyzer.extract_headings(content)
        
        # LLM analysis
        llm_analysis = await self.llm_analyzer.analyze_style(content)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(structure_analysis, llm_analysis)
        
        # Create comprehensive style profile
        style_profile = StyleProfile(
            document_id=metadata.document_id,
            file_name=metadata.file_name,
            language=metadata.language,
            
            # LLM features
            tone=llm_analysis.get('tone', 'professional'),
            formality_score=float(llm_analysis.get('formality_score', 7.0)),
            technical_level=llm_analysis.get('technical_level', 'intermediate'),
            common_phrases=llm_analysis.get('common_phrases', []),
            sentence_starters=llm_analysis.get('sentence_starters', []),
            transition_words=llm_analysis.get('transition_words', []),
            citation_style=llm_analysis.get('citation_style', 'academic'),
            
            # Rule-based features
            avg_sentence_length=structure_analysis['avg_sentence_length'],
            avg_paragraph_length=structure_analysis['avg_paragraph_length'],
            readability_score=structure_analysis['readability_score'],
            passive_voice_ratio=structure_analysis['passive_voice_ratio'],
            question_ratio=structure_analysis['question_ratio'],
            exclamation_ratio=structure_analysis['exclamation_ratio'],
            
            # Structure features
            heading_patterns=headings,
            list_usage=structure_analysis['list_usage'],
            bullet_point_usage=structure_analysis['bullet_point_usage'],
            
            # Vocabulary features
            technical_terms=vocabulary_analysis['technical_terms'],
            action_verbs=vocabulary_analysis['action_verbs'],
            sentiment_words=vocabulary_analysis['transition_words'],
            
            # Metadata
            analyzed_at=datetime.now(),
            confidence_score=confidence_score
        )
        
        self.logger.info(f"Style analysis completed for {metadata.file_name} "
                        f"(confidence: {confidence_score:.2f})")
        
        return style_profile
    
    def _calculate_confidence(self, structure_analysis: Dict[str, Any], 
                            llm_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis."""
        confidence_factors = []
        
        # Word count factor
        word_count = structure_analysis.get('total_words', 0)
        word_factor = min(1.0, word_count / 1000)  # Full confidence at 1000+ words
        confidence_factors.append(word_factor)
        
        # Sentence count factor
        sentence_count = structure_analysis.get('total_sentences', 0)
        sentence_factor = min(1.0, sentence_count / 50)  # Full confidence at 50+ sentences
        confidence_factors.append(sentence_factor)
        
        # LLM response completeness
        expected_fields = ['tone', 'formality_score', 'technical_level', 'common_phrases']
        llm_completeness = sum(1 for field in expected_fields if field in llm_analysis) / len(expected_fields)
        confidence_factors.append(llm_completeness)
        
        # Average confidence
        return statistics.mean(confidence_factors)


class StyleProfileManager:
    """Manages style profiles for multiple documents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzer = StyleAnalyzer()
        self.profiles: Dict[str, StyleProfile] = {}
    
    async def analyze_documents(self, documents: List[Tuple[str, DocumentMetadata]]) -> Dict[str, StyleProfile]:
        """Analyze multiple documents and return style profiles."""
        self.logger.info(f"Analyzing {len(documents)} documents for style patterns")
        
        profiles = {}
        
        for content, metadata in documents:
            try:
                profile = await self.analyzer.analyze_document(content, metadata)
                profiles[metadata.document_id] = profile
                self.profiles[metadata.document_id] = profile
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {metadata.file_name}: {e}")
        
        return profiles
    
    def get_aggregate_style(self, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get aggregated style patterns from multiple documents."""
        if document_ids is None:
            document_ids = list(self.profiles.keys())
        
        relevant_profiles = [self.profiles[doc_id] for doc_id in document_ids 
                           if doc_id in self.profiles]
        
        if not relevant_profiles:
            return {}
        
        # Aggregate numeric metrics
        avg_sentence_length = statistics.mean([p.avg_sentence_length for p in relevant_profiles])
        avg_readability = statistics.mean([p.readability_score for p in relevant_profiles])
        avg_formality = statistics.mean([p.formality_score for p in relevant_profiles])
        
        # Aggregate categorical features
        all_phrases = []
        all_starters = []
        all_technical_terms = []
        
        for profile in relevant_profiles:
            all_phrases.extend(profile.common_phrases)
            all_starters.extend(profile.sentence_starters)
            all_technical_terms.extend(profile.technical_terms)
        
        # Get most common items
        from collections import Counter
        top_phrases = [item for item, count in Counter(all_phrases).most_common(10)]
        top_starters = [item for item, count in Counter(all_starters).most_common(10)]
        top_technical = [item for item, count in Counter(all_technical_terms).most_common(10)]
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_readability_score': avg_readability,
            'avg_formality_score': avg_formality,
            'common_phrases': top_phrases,
            'sentence_starters': top_starters,
            'technical_terms': top_technical,
            'document_count': len(relevant_profiles)
        }
    
    def save_profiles(self, filepath: str) -> None:
        """Save all profiles to JSON file."""
        profiles_data = {doc_id: profile.to_dict() 
                        for doc_id, profile in self.profiles.items()}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(profiles_data)} style profiles to {filepath}")
    
    def load_profiles(self, filepath: str) -> None:
        """Load profiles from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            profiles_data = json.load(f)
        
        # Convert back to StyleProfile objects
        for doc_id, profile_data in profiles_data.items():
            # Convert datetime string back to datetime object
            profile_data['analyzed_at'] = datetime.fromisoformat(profile_data['analyzed_at'])
            
            # Create StyleProfile object
            profile = StyleProfile(**profile_data)
            self.profiles[doc_id] = profile
        
        self.logger.info(f"Loaded {len(profiles_data)} style profiles from {filepath}")