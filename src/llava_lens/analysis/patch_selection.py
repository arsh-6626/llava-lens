from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger
from llava_lens.analysis.logit_lens import LogitLensResult

logger = get_logger(__name__)


@dataclass
class PatchSelectionResult:
    high_confidence_indices: List[int]
    low_confidence_indices: List[int]
    selected_indices: List[int]
    confidences: Dict[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatchSelector:
    def __init__(self, config: Config):
        self.config = config
        self.high_threshold = config.analysis.patch_selection.high_confidence_threshold
        self.neighbor_context = config.analysis.patch_selection.neighbor_context
        self.min_patches = config.analysis.patch_selection.min_patches

    def select(
        self,
        logit_lens_result: LogitLensResult,
        layer_idx: int = -1,
    ) -> PatchSelectionResult:
        if layer_idx < 0:
            layer_idx = logit_lens_result.num_layers + layer_idx

        layer = logit_lens_result.layer_predictions[layer_idx]
        image_mask = logit_lens_result.image_token_mask

        high_conf = []
        low_conf = []
        confidences = {}

        for idx, (is_image, predictions) in enumerate(zip(image_mask, layer.predictions)):
            if not is_image:
                continue
            if predictions:
                conf = predictions[0].probability
                confidences[idx] = conf
                if conf >= self.high_threshold:
                    high_conf.append(idx)
                else:
                    low_conf.append(idx)

        selected = self._add_neighbors(high_conf, len(image_mask))

        if len(selected) < self.min_patches:
            all_image_indices = [i for i, m in enumerate(image_mask) if m]
            sorted_by_conf = sorted(
                [(i, confidences.get(i, 0)) for i in all_image_indices],
                key=lambda x: x[1],
                reverse=True,
            )
            selected = list(dict.fromkeys(selected + [i for i, _ in sorted_by_conf[:self.min_patches]]))

        return PatchSelectionResult(
            high_confidence_indices=high_conf,
            low_confidence_indices=low_conf,
            selected_indices=sorted(selected),
            confidences=confidences,
            metadata={
                "layer_idx": layer_idx,
                "high_threshold": self.high_threshold,
                "neighbor_context": self.neighbor_context,
            },
        )

    def _add_neighbors(self, indices: List[int], total: int) -> List[int]:
        expanded = set(indices)
        for idx in indices:
            for offset in range(1, self.neighbor_context + 1):
                left = idx - offset
                right = idx + offset
                if 0 <= left < total:
                    expanded.add(left)
                if 0 <= right < total:
                    expanded.add(right)
        return sorted(expanded)

    def get_patch_grid(
        self, result: PatchSelectionResult, grid_size: int
    ) -> List[List[Optional[float]]]:
        grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        for idx in result.confidences:
            image_positions = [
                i for i, is_img in enumerate(result.confidences) if is_img is not None
            ]
            if idx in result.confidences:
                pos = idx
                row = pos // grid_size
                col = pos % grid_size
                if row < grid_size and col < grid_size:
                    grid[row][col] = result.confidences[idx]
        return grid

    def get_selected_patches_info(
        self, result: PatchSelectionResult, token_labels: List[str], grid_size: int
    ) -> List[Dict[str, Any]]:
        patches = []
        for idx in result.selected_indices:
            if idx < len(token_labels):
                label = token_labels[idx]
                pos = idx
                row = pos // grid_size
                col = pos % grid_size
                patches.append({
                    "index": idx,
                    "label": label,
                    "position": (row, col),
                    "confidence": result.confidences.get(idx, 0),
                    "is_high_confidence": idx in result.high_confidence_indices,
                })
        return patches
