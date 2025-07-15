"""
Configuration management for Erguvan AI Content Generator
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration settings."""
    provider: str
    generation_model: str
    analysis_model: str
    embedding_model: str
    temperature: float
    max_tokens: int
    max_cost_per_request: float
    rate_limit_requests_per_minute: int
    

@dataclass
class BrandConfig:
    """Brand voice configuration."""
    primary_tone: str
    secondary_traits: list
    avoid: list
    preferred_sentence_length: str
    flesch_score_min: int


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self._models_config = None
        self._brand_config = None
        
    def load_models_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML."""
        if self._models_config is None:
            config_path = self.config_dir / "models.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                self._models_config = yaml.safe_load(f)
        return self._models_config
    
    def load_brand_config(self) -> Dict[str, Any]:
        """Load brand configuration from YAML."""
        if self._brand_config is None:
            config_path = self.config_dir / "brand_guide.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                self._brand_config = yaml.safe_load(f)
        return self._brand_config
    
    def get_model_config(self, use_fallback: bool = False) -> ModelConfig:
        """Get model configuration object."""
        config = self.load_models_config()
        
        if use_fallback:
            model_section = config["models"]["fallback"]
        else:
            model_section = config["models"]["primary"]
            
        settings = config["models"]["settings"]
        cost_mgmt = config["models"]["cost_management"]
        
        return ModelConfig(
            provider=model_section["provider"],
            generation_model=model_section["generation_model"],
            analysis_model=model_section["analysis_model"],
            embedding_model=model_section.get("embedding_model", "text-embedding-3-large"),
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
            max_cost_per_request=cost_mgmt["max_cost_per_request"],
            rate_limit_requests_per_minute=cost_mgmt["rate_limit_requests_per_minute"]
        )
    
    def get_brand_config(self) -> BrandConfig:
        """Get brand configuration object."""
        config = self.load_brand_config()
        
        return BrandConfig(
            primary_tone=config["voice_characteristics"]["primary_tone"],
            secondary_traits=config["voice_characteristics"]["secondary_traits"],
            avoid=config["voice_characteristics"]["avoid"],
            preferred_sentence_length=config["writing_style"]["sentence_structure"]["preferred_length"],
            flesch_score_min=int(config["quality_standards"]["readability"]["flesch_score"].replace(">= ", ""))
        )
    
    def validate_environment(self) -> bool:
        """Validate required environment variables."""
        config = self.load_models_config()
        required_vars = config["required_env_vars"]
        
        missing = []
        for var in required_vars:
            if var not in os.environ:
                missing.append(var)
        
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {missing}")
        
        return True


# Global configuration instance
config_manager = ConfigManager()