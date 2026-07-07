import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger

logger = get_logger(__name__)


class LocalStorage:
    def __init__(self, config: Config):
        self.config = config
        self.base_dir = Path(config.storage.base_dir)
        self.raw_dir = self.base_dir / config.storage.raw_dir
        self.processed_dir = self.base_dir / config.storage.processed_dir
        self.embeddings_dir = self.base_dir / config.storage.embeddings_dir
        self.results_dir = self.base_dir / config.storage.results_dir
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in [self.raw_dir, self.processed_dir, self.embeddings_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_path(self, subdir: str, filename: str) -> Path:
        base = self.base_dir / subdir
        base.mkdir(parents=True, exist_ok=True)
        return base / filename

    def save_image(self, image_path: Union[str, Path], subdir: str = "raw") -> Path:
        src = Path(image_path)
        dst = self.get_path(subdir, src.name)
        shutil.copy2(src, dst)
        logger.info(f"Saved image to {dst}")
        return dst

    def save_json(self, data: Any, filename: str, subdir: str = "results") -> Path:
        path = self.get_path(subdir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON to {path}")
        return path

    def load_json(self, filename: str, subdir: str = "results") -> Any:
        path = self.get_path(subdir, filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path) as f:
            return json.load(f)

    def save_embedding(self, embedding: Any, filename: str) -> Path:
        import torch
        path = self.embeddings_dir / filename
        torch.save(embedding, path)
        logger.info(f"Saved embedding to {path}")
        return path

    def load_embedding(self, filename: str) -> Any:
        import torch
        path = self.embeddings_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Embedding not found: {path}")
        return torch.load(path)

    def save_html(self, content: str, filename: str) -> Path:
        path = self.results_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved HTML to {path}")
        return path

    def list_files(self, subdir: str, extensions: Optional[List[str]] = None) -> List[Path]:
        base = self.base_dir / subdir
        if not base.exists():
            return []
        files = list(base.iterdir())
        if extensions:
            files = [f for f in files if f.suffix.lower() in extensions]
        return sorted(files)

    def get_experiment_dir(self, experiment_name: str) -> Path:
        exp_dir = self.results_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def cleanup(self, subdir: Optional[str] = None) -> None:
        if subdir:
            target = self.base_dir / subdir
            if target.exists():
                shutil.rmtree(target)
                logger.info(f"Cleaned up {target}")
        else:
            for d in [self.raw_dir, self.processed_dir, self.embeddings_dir, self.results_dir]:
                if d.exists():
                    shutil.rmtree(d)
                    d.mkdir(parents=True, exist_ok=True)
            logger.info("Cleaned up all storage directories")
