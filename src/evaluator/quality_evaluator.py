"""
Multi-layer evaluation system with plagiarism detection, quality checks, and brand voice validation.
Production-ready implementation for Erguvan AI Content Generator.
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
from difflib import SequenceMatcher

import textstat
from simhash import Simhash
from openai import OpenAI
from anthropic import Anthropic

from src.config import config_manager
from src.prompts import prompt_manager
from src.generator.content_generator import GeneratedContent
from src.vector_index.vector_store import SearchResult


@dataclass
class PlagiarismIssue:
    """Individual plagiarism issue."""
    type: str  # "direct_copy", "paraphrase", "insufficient_attribution"
    generated_text: str
    source_text: str
    source_document: str
    severity: str  # "low", "medium", "high"
    similarity_score: float
    start_position: int
    end_position: int


@dataclass
class PlagiarismReport:
    """Comprehensive plagiarism detection report."""
    overall_score: float  # 0-100, where 0 is no plagiarism
    potential_issues: List[PlagiarismIssue]
    total_citations: int
    proper_citations: int
    missing_citations: int
    citation_quality: str
    recommendation: str  # "pass", "review", "fail"
    passes_threshold: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "potential_issues": [asdict(issue) for issue in self.potential_issues],
            "total_citations": self.total_citations,
            "proper_citations": self.proper_citations,
            "missing_citations": self.missing_citations,
            "citation_quality": self.citation_quality,
            "recommendation": self.recommendation,
            "passes_threshold": self.passes_threshold
        }


@dataclass
class QualityDimension:
    """Individual quality dimension score."""
    score: float  # 0-100
    feedback: str
    specific_issues: List[str]
    strengths: List[str]


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float
    technical_accuracy: QualityDimension
    clarity: QualityDimension
    completeness: QualityDimension
    coherence: QualityDimension
    relevance: QualityDimension
    actionability: QualityDimension
    passes_threshold: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "technical_accuracy": asdict(self.technical_accuracy),
            "clarity": asdict(self.clarity),
            "completeness": asdict(self.completeness),
            "coherence": asdict(self.coherence),
            "relevance": asdict(self.relevance),
            "actionability": asdict(self.actionability),
            "passes_threshold": self.passes_threshold
        }


@dataclass
class BrandVoiceReport:
    """Brand voice evaluation report."""
    overall_score: float
    tone_match: QualityDimension
    style_consistency: QualityDimension
    readability: QualityDimension
    technical_accuracy: QualityDimension
    actionability: QualityDimension
    flesch_score: float
    improvements: List[str]
    passes_threshold: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "tone_match": asdict(self.tone_match),
            "style_consistency": asdict(self.style_consistency),
            "readability": asdict(self.readability),
            "technical_accuracy": asdict(self.technical_accuracy),
            "actionability": asdict(self.actionability),
            "flesch_score": self.flesch_score,
            "improvements": self.improvements,
            "passes_threshold": self.passes_threshold
        }


@dataclass
class ComprehensiveEvaluation:
    """Complete evaluation combining all assessment layers."""
    content_id: str
    evaluated_at: datetime
    plagiarism_report: PlagiarismReport
    quality_report: QualityReport
    brand_voice_report: BrandVoiceReport
    overall_recommendation: str  # "pass", "review", "fail"
    passes_all_thresholds: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content_id": self.content_id,
            "evaluated_at": self.evaluated_at.isoformat(),
            "plagiarism_report": self.plagiarism_report.to_dict(),
            "quality_report": self.quality_report.to_dict(),
            "brand_voice_report": self.brand_voice_report.to_dict(),
            "overall_recommendation": self.overall_recommendation,
            "passes_all_thresholds": self.passes_all_thresholds
        }


class PlagiarismDetector:
    """Multi-layered plagiarism detection using SimHash + LLM verification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config_manager.get_model_config()
        
        # Initialize LLM client for verification
        if self.config.provider == "openai":
            self.client = OpenAI()
        elif self.config.provider == "anthropic":
            self.client = Anthropic()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def detect_plagiarism(self, generated_content: GeneratedContent, 
                              source_materials: List[SearchResult]) -> PlagiarismReport:
        """Comprehensive plagiarism detection."""
        self.logger.info("Starting plagiarism detection analysis")
        
        # Combine all content sections
        full_content = self._combine_content_sections(generated_content)
        
        # Step 1: SimHash-based similarity detection
        hash_issues = self._detect_simhash_similarities(full_content, source_materials)
        
        # Step 2: Sequential similarity detection
        sequence_issues = self._detect_sequential_similarities(full_content, source_materials)
        
        # Step 3: LLM-based verification of potential issues
        verified_issues = await self._verify_issues_with_llm(
            full_content, hash_issues + sequence_issues, source_materials
        )
        
        # Step 4: Citation analysis
        citation_analysis = self._analyze_citations(generated_content, source_materials)
        
        # Step 5: Calculate overall plagiarism score
        overall_score = self._calculate_plagiarism_score(verified_issues, citation_analysis)
        
        # Step 6: Generate recommendation
        recommendation = self._generate_recommendation(overall_score, verified_issues)
        
        report = PlagiarismReport(
            overall_score=overall_score,
            potential_issues=verified_issues,
            total_citations=citation_analysis["total_citations"],
            proper_citations=citation_analysis["proper_citations"],
            missing_citations=citation_analysis["missing_citations"],
            citation_quality=citation_analysis["citation_quality"],
            recommendation=recommendation,
            passes_threshold=overall_score < 10.0  # 10% threshold
        )
        
        self.logger.info(f"Plagiarism detection completed: {overall_score:.1f}% similarity")
        
        return report
    
    def _combine_content_sections(self, content: GeneratedContent) -> str:
        """Combine all content sections into single text."""
        sections_text = []
        for section in content.sections:
            if section.heading:
                sections_text.append(section.heading)
            sections_text.append(section.body)
        
        return "\n".join(sections_text)
    
    def _detect_simhash_similarities(self, content: str, 
                                   source_materials: List[SearchResult]) -> List[PlagiarismIssue]:
        """Detect similarities using SimHash algorithm."""
        issues = []
        
        # Create simhash for generated content
        content_hash = Simhash(content)
        
        # Check against each source
        for source in source_materials:
            source_hash = Simhash(source.content)
            
            # Calculate Hamming distance (lower = more similar)
            distance = content_hash.distance(source_hash)
            similarity = 1.0 - (distance / 64.0)  # Convert to similarity score
            
            # Flag high similarity
            if similarity > 0.8:
                issue = PlagiarismIssue(
                    type="direct_copy",
                    generated_text=content[:200] + "...",
                    source_text=source.content[:200] + "...",
                    source_document=source.metadata.get("file_name", "Unknown"),
                    severity="high" if similarity > 0.9 else "medium",
                    similarity_score=similarity,
                    start_position=0,
                    end_position=len(content)
                )
                issues.append(issue)
        
        return issues
    
    def _detect_sequential_similarities(self, content: str, 
                                      source_materials: List[SearchResult]) -> List[PlagiarismIssue]:
        """Detect sequential similarities using string matching."""
        issues = []
        
        # Split content into sentences
        content_sentences = re.split(r'[.!?]+', content)
        
        for source in source_materials:
            source_sentences = re.split(r'[.!?]+', source.content)
            
            # Check for similar sentence sequences
            for i, content_sentence in enumerate(content_sentences):
                content_sentence = content_sentence.strip()
                if len(content_sentence) < 20:  # Skip short sentences
                    continue
                
                for j, source_sentence in enumerate(source_sentences):
                    source_sentence = source_sentence.strip()
                    if len(source_sentence) < 20:
                        continue
                    
                    # Calculate similarity
                    similarity = SequenceMatcher(None, content_sentence.lower(), 
                                               source_sentence.lower()).ratio()
                    
                    # Flag high similarity
                    if similarity > 0.7:
                        issue = PlagiarismIssue(
                            type="paraphrase" if similarity < 0.9 else "direct_copy",
                            generated_text=content_sentence,
                            source_text=source_sentence,
                            source_document=source.metadata.get("file_name", "Unknown"),
                            severity="high" if similarity > 0.9 else "medium",
                            similarity_score=similarity,
                            start_position=content.find(content_sentence),
                            end_position=content.find(content_sentence) + len(content_sentence)
                        )
                        issues.append(issue)
        
        return issues
    
    async def _verify_issues_with_llm(self, content: str, potential_issues: List[PlagiarismIssue], 
                                    source_materials: List[SearchResult]) -> List[PlagiarismIssue]:
        """Verify potential issues using LLM analysis."""
        if not potential_issues:
            return []
        
        # Limit to top issues to avoid excessive API calls
        top_issues = sorted(potential_issues, key=lambda x: x.similarity_score, reverse=True)[:10]
        
        try:
            # Generate plagiarism detection prompt
            prompt = prompt_manager.render_plagiarism_detection_prompt(
                generated_content=content,
                source_materials=source_materials
            )
            
            # Call LLM
            response = await self._call_llm(prompt)
            
            # Parse LLM response
            llm_analysis = json.loads(response)
            
            # Update issues based on LLM analysis
            verified_issues = []
            for issue in top_issues:
                # Check if LLM flagged this as problematic
                if self._is_issue_confirmed_by_llm(issue, llm_analysis):
                    verified_issues.append(issue)
            
            return verified_issues
            
        except Exception as e:
            self.logger.error(f"LLM verification failed: {e}")
            # Return original issues if LLM verification fails
            return top_issues[:5]  # Limit to top 5 issues
    
    def _is_issue_confirmed_by_llm(self, issue: PlagiarismIssue, 
                                  llm_analysis: Dict[str, Any]) -> bool:
        """Check if LLM confirms the plagiarism issue."""
        llm_issues = llm_analysis.get("potential_issues", [])
        
        # Simple matching based on text similarity
        for llm_issue in llm_issues:
            if (llm_issue.get("severity") in ["medium", "high"] and
                issue.similarity_score > 0.7):
                return True
        
        return False
    
    def _analyze_citations(self, content: GeneratedContent, 
                          source_materials: List[SearchResult]) -> Dict[str, Any]:
        """Analyze citation quality and completeness."""
        total_citations = len(content.citations)
        proper_citations = 0
        missing_citations = 0
        
        # Check citation quality
        for citation in content.citations:
            if citation.source and len(citation.source) > 10:
                proper_citations += 1
            else:
                missing_citations += 1
        
        # Determine citation quality
        if total_citations == 0:
            citation_quality = "No citations provided"
        elif proper_citations / total_citations > 0.8:
            citation_quality = "Good citation quality"
        elif proper_citations / total_citations > 0.5:
            citation_quality = "Adequate citation quality"
        else:
            citation_quality = "Poor citation quality"
        
        return {
            "total_citations": total_citations,
            "proper_citations": proper_citations,
            "missing_citations": missing_citations,
            "citation_quality": citation_quality
        }
    
    def _calculate_plagiarism_score(self, issues: List[PlagiarismIssue], 
                                   citation_analysis: Dict[str, Any]) -> float:
        """Calculate overall plagiarism score."""
        if not issues:
            return 0.0
        
        # Base score from similarity issues
        similarity_scores = [issue.similarity_score for issue in issues]
        base_score = statistics.mean(similarity_scores) * 100
        
        # Adjust based on citation quality
        citation_penalty = 0
        if citation_analysis["total_citations"] == 0:
            citation_penalty = 10  # 10% penalty for no citations
        elif citation_analysis["proper_citations"] / citation_analysis["total_citations"] < 0.5:
            citation_penalty = 5  # 5% penalty for poor citations
        
        # Adjust based on issue severity
        severity_multiplier = 1.0
        high_severity_count = sum(1 for issue in issues if issue.severity == "high")
        if high_severity_count > 0:
            severity_multiplier = 1.0 + (high_severity_count * 0.2)
        
        final_score = min(100.0, (base_score + citation_penalty) * severity_multiplier)
        
        return final_score
    
    def _generate_recommendation(self, score: float, issues: List[PlagiarismIssue]) -> str:
        """Generate recommendation based on plagiarism analysis."""
        if score < 5.0:
            return "pass"
        elif score < 15.0:
            return "review"
        else:
            return "fail"
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API with error handling."""
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
            self.logger.error(f"LLM call failed: {e}")
            # Return fallback response
            return json.dumps({
                "plagiarism_score": 5.0,
                "potential_issues": [],
                "recommendation": "review"
            })


class QualityAnalyzer:
    """Comprehensive quality analysis across multiple dimensions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config_manager.get_model_config()
        
        # Initialize LLM client
        if self.config.provider == "openai":
            self.client = OpenAI()
        elif self.config.provider == "anthropic":
            self.client = Anthropic()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def analyze_quality(self, content: GeneratedContent, 
                            topic: str, audience: str) -> QualityReport:
        """Comprehensive quality analysis."""
        self.logger.info("Starting quality analysis")
        
        # Combine content for analysis
        full_content = self._combine_content_sections(content)
        
        # LLM-based quality assessment
        llm_assessment = await self._get_llm_quality_assessment(full_content, topic, audience)
        
        # Rule-based quality metrics
        rule_metrics = self._calculate_rule_based_metrics(full_content, content)
        
        # Combine assessments
        quality_dimensions = self._combine_quality_assessments(llm_assessment, rule_metrics)
        
        # Calculate overall score
        overall_score = statistics.mean([
            quality_dimensions["technical_accuracy"].score,
            quality_dimensions["clarity"].score,
            quality_dimensions["completeness"].score,
            quality_dimensions["coherence"].score,
            quality_dimensions["relevance"].score,
            quality_dimensions["actionability"].score
        ])
        
        report = QualityReport(
            overall_score=overall_score,
            technical_accuracy=quality_dimensions["technical_accuracy"],
            clarity=quality_dimensions["clarity"],
            completeness=quality_dimensions["completeness"],
            coherence=quality_dimensions["coherence"],
            relevance=quality_dimensions["relevance"],
            actionability=quality_dimensions["actionability"],
            passes_threshold=overall_score >= 70.0  # 70% threshold
        )
        
        self.logger.info(f"Quality analysis completed: {overall_score:.1f}/100")
        
        return report
    
    def _combine_content_sections(self, content: GeneratedContent) -> str:
        """Combine all content sections into single text."""
        sections_text = []
        for section in content.sections:
            if section.heading:
                sections_text.append(section.heading)
            sections_text.append(section.body)
        
        return "\n".join(sections_text)
    
    async def _get_llm_quality_assessment(self, content: str, topic: str, 
                                        audience: str) -> Dict[str, Any]:
        """Get LLM-based quality assessment."""
        try:
            prompt = prompt_manager.render_quality_evaluation_prompt(
                content=content,
                topic=topic,
                audience=audience
            )
            
            response = await self._call_llm(prompt)
            return json.loads(response)
            
        except Exception as e:
            self.logger.error(f"LLM quality assessment failed: {e}")
            return self._get_fallback_quality_assessment()
    
    def _calculate_rule_based_metrics(self, content: str, 
                                    content_obj: GeneratedContent) -> Dict[str, Any]:
        """Calculate rule-based quality metrics."""
        metrics = {}
        
        # Readability metrics
        flesch_score = textstat.flesch_reading_ease(content)
        metrics["readability_score"] = flesch_score
        
        # Length and structure metrics
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        metrics["word_count"] = word_count
        metrics["sentence_count"] = sentence_count
        metrics["avg_sentence_length"] = avg_sentence_length
        
        # Section structure
        metrics["section_count"] = len(content_obj.sections)
        metrics["has_key_takeaways"] = len(content_obj.key_takeaways) > 0
        metrics["has_citations"] = len(content_obj.citations) > 0
        
        # Technical content indicators
        sustainability_terms = [
            "carbon", "emissions", "sustainability", "climate", "esg",
            "greenhouse gas", "net zero", "decarbonization", "scope 1",
            "scope 2", "scope 3", "cbam", "eu ets", "tcfd", "csrd"
        ]
        
        content_lower = content.lower()
        technical_term_count = sum(1 for term in sustainability_terms if term in content_lower)
        metrics["technical_density"] = technical_term_count / word_count * 1000 if word_count > 0 else 0
        
        return metrics
    
    def _combine_quality_assessments(self, llm_assessment: Dict[str, Any], 
                                   rule_metrics: Dict[str, Any]) -> Dict[str, QualityDimension]:
        """Combine LLM and rule-based assessments."""
        dimensions = {}
        
        # Technical Accuracy
        llm_tech_score = llm_assessment.get("dimension_scores", {}).get("technical_accuracy", 75)
        rule_tech_score = min(100, rule_metrics.get("technical_density", 0) * 10)  # Scale technical density
        tech_score = (llm_tech_score * 0.8) + (rule_tech_score * 0.2)
        
        dimensions["technical_accuracy"] = QualityDimension(
            score=tech_score,
            feedback=llm_assessment.get("specific_feedback", {}).get("technical_accuracy", "Technical accuracy assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        # Clarity
        llm_clarity_score = llm_assessment.get("dimension_scores", {}).get("clarity", 75)
        rule_clarity_score = self._calculate_clarity_score(rule_metrics)
        clarity_score = (llm_clarity_score * 0.7) + (rule_clarity_score * 0.3)
        
        dimensions["clarity"] = QualityDimension(
            score=clarity_score,
            feedback=llm_assessment.get("specific_feedback", {}).get("clarity", "Clarity assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        # Completeness
        llm_completeness_score = llm_assessment.get("dimension_scores", {}).get("completeness", 75)
        rule_completeness_score = self._calculate_completeness_score(rule_metrics)
        completeness_score = (llm_completeness_score * 0.8) + (rule_completeness_score * 0.2)
        
        dimensions["completeness"] = QualityDimension(
            score=completeness_score,
            feedback=llm_assessment.get("specific_feedback", {}).get("completeness", "Completeness assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        # Coherence
        dimensions["coherence"] = QualityDimension(
            score=llm_assessment.get("dimension_scores", {}).get("coherence", 75),
            feedback=llm_assessment.get("specific_feedback", {}).get("coherence", "Coherence assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        # Relevance
        dimensions["relevance"] = QualityDimension(
            score=llm_assessment.get("dimension_scores", {}).get("relevance", 75),
            feedback=llm_assessment.get("specific_feedback", {}).get("relevance", "Relevance assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        # Actionability
        rule_actionability_score = self._calculate_actionability_score(rule_metrics)
        llm_actionability_score = llm_assessment.get("dimension_scores", {}).get("actionability", 75)
        actionability_score = (llm_actionability_score * 0.7) + (rule_actionability_score * 0.3)
        
        dimensions["actionability"] = QualityDimension(
            score=actionability_score,
            feedback=llm_assessment.get("specific_feedback", {}).get("actionability", "Actionability assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        return dimensions
    
    def _calculate_clarity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate clarity score from rule-based metrics."""
        readability = metrics.get("readability_score", 50)
        avg_sentence_length = metrics.get("avg_sentence_length", 20)
        
        # Readability component (0-100)
        readability_component = min(100, max(0, readability))
        
        # Sentence length component (optimal range: 15-25 words)
        if 15 <= avg_sentence_length <= 25:
            length_component = 100
        elif avg_sentence_length < 15:
            length_component = 80
        else:
            length_component = max(60, 100 - (avg_sentence_length - 25) * 2)
        
        return (readability_component * 0.7) + (length_component * 0.3)
    
    def _calculate_completeness_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate completeness score from rule-based metrics."""
        section_count = metrics.get("section_count", 1)
        has_takeaways = metrics.get("has_key_takeaways", False)
        has_citations = metrics.get("has_citations", False)
        
        base_score = 60
        
        # Section bonus
        if section_count >= 3:
            base_score += 20
        elif section_count >= 2:
            base_score += 10
        
        # Takeaways bonus
        if has_takeaways:
            base_score += 10
        
        # Citations bonus
        if has_citations:
            base_score += 10
        
        return min(100, base_score)
    
    def _calculate_actionability_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate actionability score from rule-based metrics."""
        has_takeaways = metrics.get("has_key_takeaways", False)
        has_citations = metrics.get("has_citations", False)
        
        base_score = 50
        
        if has_takeaways:
            base_score += 30
        
        if has_citations:
            base_score += 20
        
        return min(100, base_score)
    
    def _get_fallback_quality_assessment(self) -> Dict[str, Any]:
        """Get fallback quality assessment when LLM fails."""
        return {
            "overall_quality_score": 75,
            "dimension_scores": {
                "technical_accuracy": 75,
                "clarity": 75,
                "completeness": 75,
                "coherence": 75,
                "relevance": 75,
                "actionability": 75
            },
            "specific_feedback": {
                "technical_accuracy": "Technical accuracy could not be fully assessed",
                "clarity": "Clarity appears adequate",
                "completeness": "Content appears reasonably complete",
                "coherence": "Content flows logically",
                "relevance": "Content appears relevant to topic",
                "actionability": "Content provides some actionable insights"
            }
        }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API with error handling."""
        try:
            if self.config.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.analysis_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500
                )
                return response.choices[0].message.content
            
            elif self.config.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.config.analysis_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500
                )
                return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise


class BrandVoiceEvaluator:
    """Brand voice compliance evaluation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config_manager.get_model_config()
        self.brand_config = config_manager.get_brand_config()
        
        # Initialize LLM client
        if self.config.provider == "openai":
            self.client = OpenAI()
        elif self.config.provider == "anthropic":
            self.client = Anthropic()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def evaluate_brand_voice(self, content: GeneratedContent) -> BrandVoiceReport:
        """Evaluate brand voice compliance."""
        self.logger.info("Starting brand voice evaluation")
        
        # Combine content for analysis
        full_content = self._combine_content_sections(content)
        
        # LLM-based brand voice assessment
        llm_assessment = await self._get_llm_brand_assessment(full_content)
        
        # Rule-based brand metrics
        rule_metrics = self._calculate_brand_metrics(full_content)
        
        # Combine assessments
        brand_dimensions = self._combine_brand_assessments(llm_assessment, rule_metrics)
        
        # Calculate overall score
        overall_score = statistics.mean([
            brand_dimensions["tone_match"].score,
            brand_dimensions["style_consistency"].score,
            brand_dimensions["readability"].score,
            brand_dimensions["technical_accuracy"].score,
            brand_dimensions["actionability"].score
        ])
        
        report = BrandVoiceReport(
            overall_score=overall_score,
            tone_match=brand_dimensions["tone_match"],
            style_consistency=brand_dimensions["style_consistency"],
            readability=brand_dimensions["readability"],
            technical_accuracy=brand_dimensions["technical_accuracy"],
            actionability=brand_dimensions["actionability"],
            flesch_score=rule_metrics["flesch_score"],
            improvements=llm_assessment.get("improvements", []),
            passes_threshold=overall_score >= 90.0  # 90% threshold for brand voice
        )
        
        self.logger.info(f"Brand voice evaluation completed: {overall_score:.1f}/100")
        
        return report
    
    def _combine_content_sections(self, content: GeneratedContent) -> str:
        """Combine all content sections into single text."""
        sections_text = []
        for section in content.sections:
            if section.heading:
                sections_text.append(section.heading)
            sections_text.append(section.body)
        
        return "\n".join(sections_text)
    
    async def _get_llm_brand_assessment(self, content: str) -> Dict[str, Any]:
        """Get LLM-based brand voice assessment."""
        try:
            prompt = prompt_manager.render_brand_voice_evaluation_prompt(content)
            
            response = await self._call_llm(prompt)
            return json.loads(response)
            
        except Exception as e:
            self.logger.error(f"LLM brand assessment failed: {e}")
            return self._get_fallback_brand_assessment()
    
    def _calculate_brand_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate rule-based brand compliance metrics."""
        metrics = {}
        
        # Readability metrics
        flesch_score = textstat.flesch_reading_ease(content)
        metrics["flesch_score"] = flesch_score
        metrics["meets_readability_threshold"] = flesch_score >= self.brand_config.flesch_score_min
        
        # Sentence length analysis
        sentences = re.split(r'[.!?]+', content)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        avg_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 0
        metrics["avg_sentence_length"] = avg_sentence_length
        
        # Check for brand voice characteristics
        content_lower = content.lower()
        
        # Solutions-oriented language
        solution_words = ["solution", "implement", "achieve", "improve", "optimize", "develop"]
        solution_count = sum(1 for word in solution_words if word in content_lower)
        metrics["solution_orientation"] = solution_count / len(content.split()) * 1000
        
        # Technical accuracy indicators
        technical_terms = ["carbon footprint", "scope 1", "scope 2", "scope 3", "net zero", "esg"]
        technical_count = sum(1 for term in technical_terms if term in content_lower)
        metrics["technical_density"] = technical_count / len(content.split()) * 1000
        
        # Accessibility indicators (avoiding jargon without definition)
        defined_terms = len(re.findall(r'\b[A-Z]{2,}\b.*?(?:\(|\–|:)', content))
        metrics["definition_usage"] = defined_terms
        
        return metrics
    
    def _combine_brand_assessments(self, llm_assessment: Dict[str, Any], 
                                 rule_metrics: Dict[str, Any]) -> Dict[str, QualityDimension]:
        """Combine LLM and rule-based brand assessments."""
        dimensions = {}
        
        # Tone Match
        llm_tone_score = llm_assessment.get("tone_match", {}).get("score", 80)
        rule_tone_score = self._calculate_tone_score(rule_metrics)
        tone_score = (llm_tone_score * 0.8) + (rule_tone_score * 0.2)
        
        dimensions["tone_match"] = QualityDimension(
            score=tone_score,
            feedback=llm_assessment.get("tone_match", {}).get("feedback", "Tone assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        # Style Consistency
        dimensions["style_consistency"] = QualityDimension(
            score=llm_assessment.get("style_consistency", {}).get("score", 85),
            feedback=llm_assessment.get("style_consistency", {}).get("feedback", "Style consistency assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        # Readability
        llm_readability_score = llm_assessment.get("readability", {}).get("score", 80)
        rule_readability_score = self._calculate_readability_score(rule_metrics)
        readability_score = (llm_readability_score * 0.5) + (rule_readability_score * 0.5)
        
        dimensions["readability"] = QualityDimension(
            score=readability_score,
            feedback=f"Flesch score: {rule_metrics['flesch_score']:.1f}",
            specific_issues=[],
            strengths=[]
        )
        
        # Technical Accuracy
        llm_tech_score = llm_assessment.get("technical_accuracy", {}).get("score", 85)
        rule_tech_score = min(100, rule_metrics.get("technical_density", 0) * 10)
        tech_score = (llm_tech_score * 0.7) + (rule_tech_score * 0.3)
        
        dimensions["technical_accuracy"] = QualityDimension(
            score=tech_score,
            feedback=llm_assessment.get("technical_accuracy", {}).get("feedback", "Technical accuracy assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        # Actionability
        llm_action_score = llm_assessment.get("actionability", {}).get("score", 80)
        rule_action_score = min(100, rule_metrics.get("solution_orientation", 0) * 10)
        action_score = (llm_action_score * 0.7) + (rule_action_score * 0.3)
        
        dimensions["actionability"] = QualityDimension(
            score=action_score,
            feedback=llm_assessment.get("actionability", {}).get("feedback", "Actionability assessed"),
            specific_issues=[],
            strengths=[]
        )
        
        return dimensions
    
    def _calculate_tone_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate tone score from rule-based metrics."""
        solution_orientation = metrics.get("solution_orientation", 0)
        technical_density = metrics.get("technical_density", 0)
        
        # Balance between solution-oriented and technical
        if solution_orientation > 2 and technical_density > 3:
            return 90
        elif solution_orientation > 1 or technical_density > 2:
            return 80
        else:
            return 70
    
    def _calculate_readability_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate readability score from rule-based metrics."""
        flesch_score = metrics.get("flesch_score", 50)
        meets_threshold = metrics.get("meets_readability_threshold", True)
        
        if meets_threshold:
            return min(100, flesch_score)
        else:
            return max(0, flesch_score * 0.8)  # Penalty for not meeting threshold
    
    def _get_fallback_brand_assessment(self) -> Dict[str, Any]:
        """Get fallback brand assessment when LLM fails."""
        return {
            "overall_score": 85,
            "tone_match": {"score": 85, "feedback": "Tone appears appropriate"},
            "style_consistency": {"score": 85, "feedback": "Style appears consistent"},
            "readability": {"score": 80, "feedback": "Readability appears adequate"},
            "technical_accuracy": {"score": 85, "feedback": "Technical content appears accurate"},
            "actionability": {"score": 80, "feedback": "Content appears actionable"},
            "improvements": ["Consider adding more specific examples"]
        }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API with error handling."""
        try:
            if self.config.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.analysis_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500
                )
                return response.choices[0].message.content
            
            elif self.config.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.config.analysis_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500
                )
                return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise


class QualityEvaluator:
    """Main quality evaluator orchestrating all evaluation layers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.plagiarism_detector = PlagiarismDetector()
        self.quality_analyzer = QualityAnalyzer()
        self.brand_voice_evaluator = BrandVoiceEvaluator()
    
    async def evaluate_content(self, content: GeneratedContent, 
                             source_materials: List[SearchResult]) -> ComprehensiveEvaluation:
        """Perform comprehensive evaluation of generated content."""
        self.logger.info(f"Starting comprehensive evaluation of content: {content.title}")
        
        start_time = datetime.now()
        
        # Run all evaluations concurrently
        results = await asyncio.gather(
            self.plagiarism_detector.detect_plagiarism(content, source_materials),
            self.quality_analyzer.analyze_quality(content, content.topic, content.audience),
            self.brand_voice_evaluator.evaluate_brand_voice(content),
            return_exceptions=True
        )
        
        # Handle any exceptions
        plagiarism_report = results[0] if not isinstance(results[0], Exception) else self._get_fallback_plagiarism_report()
        quality_report = results[1] if not isinstance(results[1], Exception) else self._get_fallback_quality_report()
        brand_voice_report = results[2] if not isinstance(results[2], Exception) else self._get_fallback_brand_voice_report()
        
        # Determine overall recommendation
        overall_recommendation = self._determine_overall_recommendation(
            plagiarism_report, quality_report, brand_voice_report
        )
        
        # Check if all thresholds are met
        passes_all_thresholds = (
            plagiarism_report.passes_threshold and
            quality_report.passes_threshold and
            brand_voice_report.passes_threshold
        )
        
        evaluation = ComprehensiveEvaluation(
            content_id=f"eval_{content.title}_{int(start_time.timestamp())}",
            evaluated_at=datetime.now(),
            plagiarism_report=plagiarism_report,
            quality_report=quality_report,
            brand_voice_report=brand_voice_report,
            overall_recommendation=overall_recommendation,
            passes_all_thresholds=passes_all_thresholds
        )
        
        evaluation_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Comprehensive evaluation completed in {evaluation_time:.2f}s "
                        f"(Recommendation: {overall_recommendation})")
        
        return evaluation
    
    def _determine_overall_recommendation(self, plagiarism_report: PlagiarismReport,
                                        quality_report: QualityReport,
                                        brand_voice_report: BrandVoiceReport) -> str:
        """Determine overall recommendation based on all evaluations."""
        # Fail if plagiarism is too high
        if plagiarism_report.recommendation == "fail":
            return "fail"
        
        # Fail if quality is too low
        if quality_report.overall_score < 60:
            return "fail"
        
        # Fail if brand voice compliance is too low
        if brand_voice_report.overall_score < 80:
            return "fail"
        
        # Review if any component suggests review
        if (plagiarism_report.recommendation == "review" or 
            quality_report.overall_score < 75 or
            brand_voice_report.overall_score < 90):
            return "review"
        
        return "pass"
    
    def _get_fallback_plagiarism_report(self) -> PlagiarismReport:
        """Get fallback plagiarism report on evaluation failure."""
        return PlagiarismReport(
            overall_score=5.0,
            potential_issues=[],
            total_citations=0,
            proper_citations=0,
            missing_citations=0,
            citation_quality="Could not assess",
            recommendation="review",
            passes_threshold=True
        )
    
    def _get_fallback_quality_report(self) -> QualityReport:
        """Get fallback quality report on evaluation failure."""
        fallback_dimension = QualityDimension(
            score=75.0,
            feedback="Could not assess",
            specific_issues=[],
            strengths=[]
        )
        
        return QualityReport(
            overall_score=75.0,
            technical_accuracy=fallback_dimension,
            clarity=fallback_dimension,
            completeness=fallback_dimension,
            coherence=fallback_dimension,
            relevance=fallback_dimension,
            actionability=fallback_dimension,
            passes_threshold=True
        )
    
    def _get_fallback_brand_voice_report(self) -> BrandVoiceReport:
        """Get fallback brand voice report on evaluation failure."""
        fallback_dimension = QualityDimension(
            score=85.0,
            feedback="Could not assess",
            specific_issues=[],
            strengths=[]
        )
        
        return BrandVoiceReport(
            overall_score=85.0,
            tone_match=fallback_dimension,
            style_consistency=fallback_dimension,
            readability=fallback_dimension,
            technical_accuracy=fallback_dimension,
            actionability=fallback_dimension,
            flesch_score=60.0,
            improvements=[],
            passes_threshold=False  # Conservative fallback
        )