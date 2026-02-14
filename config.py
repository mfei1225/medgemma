from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration settings for MedGemma backend."""
    
    # Model Configuration
    model_id: str = "google/medgemma-1.5-4b-it"
    use_quantization: bool = False  # Set to True to use 8-bit quantization
    cache_dir: Optional[str] = None  # HuggingFace cache directory
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Generation Parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Hugging Face Authentication
    huggingface_token: Optional[str] = None
    
    # Device Configuration
    device: str = "cuda"  # or "cpu"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
