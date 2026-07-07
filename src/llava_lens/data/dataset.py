from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from llava_lens.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


class ImageDataset:
    def __init__(
        self,
        root_dir: Union[str, Path],
        prompt: str = "Describe the image.",
        transform: Optional[Callable] = None,
        extensions: Optional[set] = None,
    ):
        self.root_dir = Path(root_dir)
        self.prompt = prompt
        self.transform = transform
        self.extensions = extensions or SUPPORTED_EXTENSIONS

        self.image_paths = self._scan_directory()
        logger.info(f"Found {len(self.image_paths)} images in {root_dir}")

    def _scan_directory(self) -> List[Path]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        paths = []
        for ext in self.extensions:
            paths.extend(self.root_dir.glob(f"**/*{ext}"))
            paths.extend(self.root_dir.glob(f"**/*{ext.upper()}"))

        return sorted(set(paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "path": str(path),
            "filename": path.name,
            "prompt": self.prompt,
        }

    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        return [self[i] for i in indices]

    def filter_by_extension(self, extensions: set) -> "ImageDataset":
        self.extensions = extensions
        self.image_paths = self._scan_directory()
        return self

    def subset(self, start: int = 0, end: Optional[int] = None) -> "ImageDataset":
        if end is None:
            end = len(self)
        new_dataset = ImageDataset.__new__(ImageDataset)
        new_dataset.root_dir = self.root_dir
        new_dataset.prompt = self.prompt
        new_dataset.transform = self.transform
        new_dataset.extensions = self.extensions
        new_dataset.image_paths = self.image_paths[start:end]
        return new_dataset


class DatasetManager:
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)

    def create_dataset(
        self,
        name: str,
        root_dir: Union[str, Path],
        prompt: str = "Describe the image.",
    ) -> ImageDataset:
        dataset = ImageDataset(root_dir=root_dir, prompt=prompt)
        logger.info(f"Created dataset '{name}' with {len(dataset)} images")
        return dataset

    def list_datasets(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        dataset_dir = self.base_dir / name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found")

        dataset = ImageDataset(root_dir=dataset_dir)
        return {
            "name": name,
            "path": str(dataset_dir),
            "num_images": len(dataset),
            "extensions": list(dataset.extensions),
        }
