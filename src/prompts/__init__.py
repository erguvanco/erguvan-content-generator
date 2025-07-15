"""
Centralized prompt templates system for Erguvan AI Content Generator.
Production-ready implementation with Jinja2 templating.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
from dataclasses import dataclass
import logging

from src.config import config_manager


@dataclass
class PromptContext:
    """Context for prompt rendering."""
    brand_voice: Dict[str, Any]
    document_content: Optional[str] = None
    topic: Optional[str] = None
    audience: Optional[str] = None
    desired_length: Optional[int] = None
    language: str = "en"
    style_override: Optional[str] = None
    context_chunks: Optional[list] = None
    style_patterns: Optional[list] = None
    timestamp: Optional[str] = None
    content_to_evaluate: Optional[str] = None
    generated_content: Optional[str] = None
    source_materials: Optional[list] = None
    content: Optional[str] = None


class PromptTemplate:
    """Individual prompt template with metadata."""
    
    def __init__(self, name: str, template: Template, description: str = ""):
        self.name = name
        self.template = template
        self.description = description
        self.usage_count = 0
        self.last_used = None
    
    def render(self, context: PromptContext) -> str:
        """Render template with context."""
        from datetime import datetime
        
        self.usage_count += 1
        self.last_used = datetime.now()
        
        # Convert context to dict for Jinja2
        context_dict = {
            'brand_voice': context.brand_voice,
            'document_content': context.document_content,
            'topic': context.topic,
            'audience': context.audience,
            'desired_length': context.desired_length,
            'language': context.language,
            'style_override': context.style_override,
            'context_chunks': context.context_chunks or [],
            'style_patterns': context.style_patterns or [],
            'timestamp': context.timestamp or datetime.now().isoformat(),
            'content_to_evaluate': context.content_to_evaluate,
            'generated_content': context.generated_content,
            'source_materials': context.source_materials or [],
            'content': context.content
        }
        
        return self.template.render(**context_dict)


class PromptManager:
    """Centralized prompt management system."""
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or Path(__file__).parent.parent.parent / "prompts"
        self.logger = logging.getLogger(__name__)
        self.templates: Dict[str, PromptTemplate] = {}
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Load brand voice configuration
        self.brand_config = config_manager.get_brand_config()
        
        # Load all prompt templates
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all prompt templates from YAML files."""
        system_prompts_file = self.prompts_dir / "system_prompts.yaml"
        
        if not system_prompts_file.exists():
            raise FileNotFoundError(f"System prompts file not found: {system_prompts_file}")
        
        with open(system_prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = yaml.safe_load(f)
        
        # Create templates for each prompt
        for prompt_name, prompt_content in prompts_data.items():
            if isinstance(prompt_content, str):
                template = self.jinja_env.from_string(prompt_content)
                self.templates[prompt_name] = PromptTemplate(
                    name=prompt_name,
                    template=template,
                    description=f"Template for {prompt_name.replace('_', ' ')}"
                )
        
        self.logger.info(f"Loaded {len(self.templates)} prompt templates")
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific prompt template."""
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        return self.templates[template_name]
    
    def render_system_prompt(self, template_name: str, **kwargs) -> str:
        """Render a system prompt with brand voice context."""
        template = self.get_template(template_name)
        
        # Create context with brand voice
        context = PromptContext(
            brand_voice={
                'primary_tone': self.brand_config.primary_tone,
                'secondary_traits': self.brand_config.secondary_traits,
                'avoid': self.brand_config.avoid,
                'preferred_sentence_length': self.brand_config.preferred_sentence_length,
                'flesch_score_min': self.brand_config.flesch_score_min
            },
            **kwargs
        )
        
        return template.render(context)
    
    def render_style_analysis_prompt(self, document_content: str) -> str:
        """Render style analysis prompt for competitor documents."""
        return self.render_system_prompt(
            'style_analysis_prompt',
            document_content=document_content
        )
    
    def render_content_generation_prompt(self, topic: str, audience: str, 
                                       desired_length: int, context_chunks: list,
                                       style_patterns: list, language: str = "en",
                                       style_override: str = None) -> str:
        """Render content generation prompt with RAG context."""
        return self.render_system_prompt(
            'content_generation_prompt',
            topic=topic,
            audience=audience,
            desired_length=desired_length,
            context_chunks=context_chunks,
            style_patterns=style_patterns,
            language=language,
            style_override=style_override
        )
    
    def render_brand_voice_evaluation_prompt(self, content_to_evaluate: str) -> str:
        """Render brand voice evaluation prompt."""
        return self.render_system_prompt(
            'brand_voice_evaluation_prompt',
            content_to_evaluate=content_to_evaluate
        )
    
    def render_plagiarism_detection_prompt(self, generated_content: str, 
                                         source_materials: list) -> str:
        """Render plagiarism detection prompt."""
        return self.render_system_prompt(
            'plagiarism_detection_prompt',
            generated_content=generated_content,
            source_materials=source_materials
        )
    
    def render_quality_evaluation_prompt(self, content: str, audience: str, 
                                       topic: str) -> str:
        """Render quality evaluation prompt."""
        return self.render_system_prompt(
            'quality_evaluation_prompt',
            content=content,
            audience=audience,
            topic=topic
        )
    
    def get_template_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all templates."""
        stats = {}
        
        for name, template in self.templates.items():
            stats[name] = {
                'usage_count': template.usage_count,
                'last_used': template.last_used.isoformat() if template.last_used else None,
                'description': template.description
            }
        
        return stats
    
    def reload_templates(self) -> None:
        """Reload all templates from disk."""
        self.templates.clear()
        self._load_templates()
        self.logger.info("Reloaded all prompt templates")


# Global prompt manager instance
prompt_manager = PromptManager()