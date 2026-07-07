from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger
from llava_lens.data.dataset import ImageDataset
from llava_lens.processing.batch import BatchProcessor, BatchResult
from llava_lens.models.base import BaseModelWrapper

logger = get_logger(__name__)


class ProcessingPipeline:
    def __init__(self, config: Config, model: BaseModelWrapper):
        self.config = config
        self.model = model
        self.batch_processor = BatchProcessor(config)

    def process_single(
        self,
        image: Any,
        prompt: str,
        analysis_fn: Callable,
    ) -> Any:
        return analysis_fn(image, prompt)

    def process_directory(
        self,
        image_dir: Union[str, Path],
        prompt: str,
        analysis_fn: Callable,
        output_dir: Optional[Union[str, Path]] = None,
        batch_size: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> BatchResult:
        dataset = ImageDataset(root_dir=image_dir, prompt=prompt)

        def process_item(item):
            result = analysis_fn(item["image"], item["prompt"])
            return {
                "path": item["path"],
                "filename": item["filename"],
                "result": result,
            }

        def progress_callback(progress, job):
            pct = int(progress * 100)
            logger.info(f"Progress: {pct}% ({job.processed_items}/{job.total_items})")

        batch_result = self.batch_processor.process_dataset(
            dataset=dataset,
            process_fn=process_item,
            batch_size=batch_size,
            max_items=max_items,
            progress_callback=progress_callback,
        )

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self._save_results(batch_result, output_path)

        return batch_result

    def process_image_list(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str,
        analysis_fn: Callable,
    ) -> BatchResult:
        items = []
        for path in image_paths:
            items.append({"path": str(path), "prompt": prompt})

        def process_item(item):
            from PIL import Image
            image = Image.open(item["path"]).convert("RGB")
            result = analysis_fn(image, item["prompt"])
            return {"path": item["path"], "result": result}

        return self.batch_processor.process_parallel(
            items=items,
            process_fn=process_item,
        )

    def _save_results(self, batch_result: BatchResult, output_dir: Path) -> None:
        import json

        summary = {
            "job_id": batch_result.job.job_id,
            "status": batch_result.job.status.value,
            "summary": batch_result.summary,
            "errors": batch_result.job.errors,
        }

        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")
