import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(config.storage.base_dir) / config.storage.cache_dir
        self.enabled = config.storage.cache_enabled
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def _get_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._get_key(key)}.json"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self._get_path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._get_path(key)
        try:
            with open(path, "w") as f:
                json.dump(value, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache entry {key}: {e}")

    def has(self, key: str) -> bool:
        if not self.enabled:
            return False
        return self._get_path(key).exists()

    def delete(self, key: str) -> None:
        path = self._get_path(key)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
        logger.info("Cache cleared")

    def get_or_set(self, key: str, compute_fn, *args, **kwargs) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute_fn(*args, **kwargs)
        self.set(key, value)
        return value
