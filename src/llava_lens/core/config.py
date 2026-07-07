"""Configuration management with YAML support and validation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs"


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "auto"
    torch_dtype: str = "float16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    image_size: int = 336
    patch_size: int = 14


class LogitLensConfig(BaseModel):
    """Logit lens analysis configuration."""

    top_k: int = 5
    confidence_threshold: float = 0.01


class AttentionConfig(BaseModel):
    """Attention analysis configuration."""

    head_aggregation: str = "mean"
    layer_aggregation: str = "last"


class PatchSelectionConfig(BaseModel):
    """Patch selection configuration."""

    high_confidence_threshold: float = 0.01
    neighbor_context: int = 2
    min_patches: int = 10


class AnalysisConfig(BaseModel):
    """Analysis configuration."""

    logit_lens: LogitLensConfig = Field(default_factory=LogitLensConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    patch_selection: PatchSelectionConfig = Field(default_factory=PatchSelectionConfig)


class ProcessingConfig(BaseModel):
    """Processing configuration."""

    batch_size: int = 1
    num_workers: int = 4
    max_queue_size: int = 100
    timeout: int = 300


class StorageConfig(BaseModel):
    """Storage configuration."""

    base_dir: str = "./data"
    raw_dir: str = "raw"
    processed_dir: str = "processed"
    embeddings_dir: str = "embeddings"
    results_dir: str = "results"
    cache_enabled: bool = True
    cache_dir: str = ".cache"


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    enabled: bool = True
    db_path: str = "./data/tracking.db"
    auto_track: bool = True


class WebConfig(BaseModel):
    """Web interface configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]
    max_upload_size: int = 52428800  # 50MB


class VisualizationConfig(BaseModel):
    """Visualization configuration."""

    template_dir: str = "./templates"
    output_format: str = "html"
    image_size: int = 336


class AppConfig(BaseModel):
    """Application configuration."""

    name: str = "llava-lens"
    version: str = "2.0.0"
    debug: bool = False


class Config(BaseModel):
    """Main configuration class."""

    app: AppConfig = Field(default_factory=AppConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    class Config:
        extra = "forbid"

    def get_storage_path(self, subdir: str) -> Path:
        """Get a storage subdirectory path."""
        base = Path(self.storage.base_dir)
        return base / subdir

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        dirs = [
            self.storage.raw_dir,
            self.storage.processed_dir,
            self.storage.embeddings_dir,
            self.storage.results_dir,
            self.storage.cache_dir,
        ]
        base = Path(self.storage.base_dir)
        for d in dirs:
            (base / d).mkdir(parents=True, exist_ok=True)


_config: Optional[Config] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = CONFIG_DIR / "default.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return Config(**data)
    return Config()


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Get or create the global configuration."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset to default configuration."""
    global _config
    _config = None
