import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger
from llava_lens.data.dataset import ImageDataset

logger = get_logger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    results: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.processed_items / self.total_items

    @property
    def elapsed_time(self) -> Optional[float]:
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def items_per_second(self) -> float:
        elapsed = self.elapsed_time
        if elapsed is None or elapsed == 0:
            return 0.0
        return self.processed_items / elapsed

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class BatchResult:
    job: BatchJob
    successful_results: List[Any] = field(default_factory=list)
    failed_indices: List[int] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.batch_size = config.processing.batch_size
        self.num_workers = config.processing.num_workers
        self.timeout = config.processing.timeout

    def process_dataset(
        self,
        dataset: ImageDataset,
        process_fn: Callable,
        batch_size: Optional[int] = None,
        max_items: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> BatchResult:
        batch_size = batch_size or self.batch_size
        total = len(dataset)
        if max_items:
            total = min(total, max_items)

        job = BatchJob(total_items=total)
        job.update(status=JobStatus.RUNNING, start_time=time.time())

        logger.info(f"Starting batch job {job.job_id}: {total} items, batch_size={batch_size}")

        results = []
        failed_indices = []

        try:
            indices = list(range(total))
            batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

            for batch_idx, batch_indices in enumerate(batches):
                batch_data = dataset.get_batch(batch_indices)

                for item_idx, item in enumerate(batch_data):
                    try:
                        result = process_fn(item)
                        results.append(result)
                        job.update(processed_items=job.processed_items + 1)
                    except Exception as e:
                        logger.error(f"Failed to process item {batch_indices[item_idx]}: {e}")
                        failed_indices.append(batch_indices[item_idx])
                        job.update(failed_items=job.failed_items + 1)
                        job.errors.append({
                            "index": batch_indices[item_idx],
                            "error": str(e),
                        })

                if progress_callback:
                    progress_callback(job.progress, job)

        except Exception as e:
            logger.error(f"Batch job failed: {e}")
            job.update(status=JobStatus.FAILED)
        else:
            job.update(status=JobStatus.COMPLETED)

        finally:
            job.update(end_time=time.time())
            job.update(results=results)

        logger.info(
            f"Batch job {job.job_id} completed: "
            f"{job.processed_items}/{total} items, "
            f"{job.failed_items} failed, "
            f"{job.elapsed_time:.2f}s"
        )

        return BatchResult(
            job=job,
            successful_results=results,
            failed_indices=failed_indices,
            summary={
                "total": total,
                "processed": job.processed_items,
                "failed": job.failed_items,
                "elapsed_time": job.elapsed_time,
                "items_per_second": job.items_per_second,
            },
        )

    def process_parallel(
        self,
        items: List[Any],
        process_fn: Callable,
        max_workers: Optional[int] = None,
    ) -> BatchResult:
        max_workers = max_workers or self.num_workers
        job = BatchJob(total_items=len(items))
        job.update(status=JobStatus.RUNNING, start_time=time.time())

        results = [None] * len(items)
        failed_indices = []

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(process_fn, item): idx for idx, item in enumerate(items)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result(timeout=self.timeout)
                        results[idx] = result
                        job.update(processed_items=job.processed_items + 1)
                    except Exception as e:
                        logger.error(f"Failed to process item {idx}: {e}")
                        failed_indices.append(idx)
                        job.update(failed_items=job.failed_items + 1)
                        job.errors.append({"index": idx, "error": str(e)})

        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            job.update(status=JobStatus.FAILED)
        else:
            job.update(status=JobStatus.COMPLETED)

        finally:
            job.update(end_time=time.time(), results=results)

        return BatchResult(
            job=job,
            successful_results=[r for r in results if r is not None],
            failed_indices=failed_indices,
            summary={
                "total": len(items),
                "processed": job.processed_items,
                "failed": job.failed_items,
                "elapsed_time": job.elapsed_time,
                "items_per_second": job.items_per_second,
            },
        )
