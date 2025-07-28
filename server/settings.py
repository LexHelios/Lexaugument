"""
LexOS Vibe Coder - Settings and Configuration
Environment variable loading and configuration management
"""
from pydantic_settings import BaseSettings
from pydantic import Field, Field
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = {"extra": "allow", "env_file": ".env"}  # Allow extra fields from .env

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="LEXOS_HOST")
    PORT: int = Field(default=8000, env="LEXOS_PORT")
    DEBUG: bool = Field(default=False, env="LEXOS_DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LEXOS_LOG_LEVEL")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="LEXOS_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="LEXOS_JWT_ALGORITHM")
    JWT_EXPIRATION_HOURS: int = Field(default=24, env="LEXOS_JWT_EXPIRATION_HOURS")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
        env="LEXOS_ALLOWED_ORIGINS"
    )
    
    # API Keys - External Services
    TOGETHER_API_KEY: Optional[str] = Field(default=None, env="TOGETHER_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    ELEVENLABS_API_KEY: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = Field(default=None, env="DEEPSEEK_API_KEY")
    DEEPGRAM_API_KEY: Optional[str] = Field(default=None, env="DEEPGRAM_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    PERPLEXITY_API_KEY: Optional[str] = Field(default=None, env="PERPLEXITY_API_KEY")
    COHERE_API_KEY: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    GEMINI_API_KEY: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    
    # vLLM Configuration
    VLLM_HOST: str = Field(default="localhost", env="VLLM_HOST")
    VLLM_PORT: int = Field(default=8001, env="VLLM_PORT")
    VLLM_MODELS: List[str] = Field(
        default=[
            "meta-llama/Llama-3-70b-chat-hf",
            "google/gemma-2-27b-it",
            "mistralai/Mixtral-8x22B-Instruct-v0.1"
        ],
        env="VLLM_MODELS"
    )
    DEFAULT_MODEL: str = Field(default="meta-llama/Llama-3-70b-chat-hf", env="VLLM_DEFAULT_MODEL")
    
    # Memory Configuration
    LMDB_PATH: str = Field(default="./data/lmdb", env="LEXOS_LMDB_PATH")
    LMDB_MAP_SIZE: int = Field(default=1024**3, env="LEXOS_LMDB_MAP_SIZE")  # 1GB
    ENCRYPTION_KEY: Optional[str] = Field(default=None, env="LEXOS_ENCRYPTION_KEY")
    
    # Vector Store Configuration (Milvus)
    MILVUS_HOST: str = Field(default="localhost", env="MILVUS_HOST")
    MILVUS_PORT: int = Field(default=19530, env="MILVUS_PORT")
    MILVUS_COLLECTION: str = Field(default="lexos_vectors", env="MILVUS_COLLECTION")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # Voice Configuration
    WHISPER_MODEL: str = Field(default="base", env="WHISPER_MODEL")
    TTS_MODEL: str = Field(default="tts_models/en/ljspeech/tacotron2-DDC", env="TTS_MODEL")
    SAMPLE_RATE: int = Field(default=16000, env="SAMPLE_RATE")
    AUDIO_CHANNELS: int = Field(default=1, env="AUDIO_CHANNELS")
    
    # Avatar Configuration
    AVATAR_MODEL_PATH: str = Field(default="./models/avatar", env="AVATAR_MODEL_PATH")
    LIPSYNC_MODEL: str = Field(default="Wav2Lip", env="LIPSYNC_MODEL")  # Wav2Lip or MuseTalk
    AVATAR_FPS: int = Field(default=30, env="AVATAR_FPS")
    
    # Health Monitoring
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # seconds
    PROMETHEUS_PORT: int = Field(default=8002, env="PROMETHEUS_PORT")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    
    # Backup Configuration
    BACKUP_PATH: str = Field(default="/mnt/nas/backups", env="BACKUP_PATH")
    BACKUP_INTERVAL_HOURS: int = Field(default=24, env="BACKUP_INTERVAL_HOURS")
    BACKUP_RETENTION_DAYS: int = Field(default=30, env="BACKUP_RETENTION_DAYS")
    
    # Agent Configuration
    AGENT_MEMORY_LIMIT: int = Field(default=10000, env="AGENT_MEMORY_LIMIT")  # messages
    AGENT_CONTEXT_WINDOW: int = Field(default=4096, env="AGENT_CONTEXT_WINDOW")  # tokens
    RAG_TOP_K: int = Field(default=5, env="RAG_TOP_K")
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.7, env="RAG_SIMILARITY_THRESHOLD")
    
    # Digital Soul Configuration
    DIGITAL_SOUL_ENABLED: bool = Field(default=True, env="DIGITAL_SOUL_ENABLED")
    WEALTH_ENGINE_ENABLED: bool = Field(default=False, env="WEALTH_ENGINE_ENABLED")
    MARKET_DATA_API_KEY: Optional[str] = Field(default=None, env="MARKET_DATA_API_KEY")
    
    # Development Configuration
    MOCK_EXTERNAL_APIS: bool = Field(default=False, env="MOCK_EXTERNAL_APIS")
    ENABLE_PROFILING: bool = Field(default=False, env="ENABLE_PROFILING")

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create necessary directories
        Path(self.LMDB_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.AVATAR_MODEL_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.BACKUP_PATH).mkdir(parents=True, exist_ok=True)
        
        # Generate encryption key if not provided
        if not self.ENCRYPTION_KEY:
            from cryptography.fernet import Fernet
            self.ENCRYPTION_KEY = Fernet.generate_key().decode()
    
    @property
    def database_url(self) -> str:
        """Get database URL for LMDB"""
        return f"lmdb://{self.LMDB_PATH}"
    
    @property
    def milvus_url(self) -> str:
        """Get Milvus connection URL"""
        return f"http://{self.MILVUS_HOST}:{self.MILVUS_PORT}"
    
    @property
    def vllm_url(self) -> str:
        """Get vLLM server URL"""
        return f"http://{self.VLLM_HOST}:{self.VLLM_PORT}"

# Global settings instance
settings = Settings()

# Validate critical settings
def validate_settings():
    """Validate critical settings and warn about missing configurations"""
    warnings = []
    
    if settings.SECRET_KEY == "your-secret-key-change-in-production":
        warnings.append("⚠️  Using default SECRET_KEY - change in production!")
    
    if not settings.TOGETHER_API_KEY and not settings.MOCK_EXTERNAL_APIS:
        warnings.append("⚠️  TOGETHER_API_KEY not set - some features may not work")
    
    if not settings.ENCRYPTION_KEY:
        warnings.append("⚠️  ENCRYPTION_KEY not set - generating temporary key")
    
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(warning)

# Validate on import
validate_settings()
