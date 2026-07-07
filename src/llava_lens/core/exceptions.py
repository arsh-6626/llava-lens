"""Custom exceptions for llava-lens."""


class LlavaLensError(Exception):
    """Base exception for llava-lens."""

    pass


class ModelNotFoundError(LlavaLensError):
    """Raised when a model is not found in the registry."""

    def __init__(self, model_name: str, available: list = None):
        msg = f"Model '{model_name}' not found"
        if available:
            msg += f". Available models: {available}"
        super().__init__(msg)
        self.model_name = model_name
        self.available = available or []


class ConfigError(LlavaLensError):
    """Raised for configuration errors."""

    pass


class ProcessingError(LlavaLensError):
    """Raised when image processing fails."""

    def __init__(self, image_path: str, reason: str):
        super().__init__(f"Failed to process '{image_path}': {reason}")
        self.image_path = image_path
        self.reason = reason


class StorageError(LlavaLensError):
    """Raised for storage-related errors."""

    pass


class AnalysisError(LlavaLensError):
    """Raised when analysis fails."""

    pass


class WebError(LlavaLensError):
    """Raised for web interface errors."""

    pass
