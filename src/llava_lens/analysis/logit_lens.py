from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenPrediction:
    token: str
    probability: float
    token_id: Optional[int] = None


@dataclass
class LayerPredictions:
    layer_index: int
    predictions: List[List[TokenPrediction]]
    image_token_predictions: Optional[List[List[TokenPrediction]]] = None


@dataclass
class LogitLensResult:
    token_labels: List[str]
    layer_predictions: List[LayerPredictions]
    image_token_mask: List[bool]
    image_size: int
    patch_size: int
    num_layers: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class LogitLensAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.top_k = config.analysis.logit_lens.top_k
        self.confidence_threshold = config.analysis.logit_lens.confidence_threshold

    def analyze(
        self,
        hidden_states: Tuple[torch.Tensor],
        norm_layer: Any,
        lm_head: Any,
        tokenizer: Any,
        input_ids: torch.Tensor,
        image_size: int = 336,
        patch_size: int = 14,
    ) -> LogitLensResult:
        token_labels = self._build_token_labels(input_ids, tokenizer, image_size, patch_size)
        image_token_mask = self._build_image_token_mask(input_ids, image_size, patch_size)
        num_layers = len(hidden_states)

        all_layer_predictions = []
        for layer_idx in range(num_layers):
            layer_hidden = hidden_states[layer_idx]
            normalized = norm_layer(layer_hidden)
            logits = lm_head(normalized)
            probs = torch.softmax(logits, dim=-1)

            top_k_probs, top_k_indices = torch.topk(probs, k=self.top_k, dim=-1)

            layer_predictions = []
            seq_len = top_k_indices.shape[1]
            for pos in range(seq_len):
                token_preds = []
                for k in range(self.top_k):
                    token_id = top_k_indices[0, pos, k].item()
                    prob = top_k_probs[0, pos, k].item()
                    token_text = tokenizer.decode([token_id])
                    token_preds.append(
                        TokenPrediction(token=token_text, probability=prob, token_id=token_id)
                    )
                layer_predictions.append(token_preds)

            all_layer_predictions.append(
                LayerPredictions(layer_index=layer_idx, predictions=layer_predictions)
            )

        return LogitLensResult(
            token_labels=token_labels,
            layer_predictions=all_layer_predictions,
            image_token_mask=image_token_mask,
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
        )

    def _build_token_labels(
        self, input_ids: torch.Tensor, tokenizer: Any, image_size: int, patch_size: int
    ) -> List[str]:
        num_patches = (image_size // patch_size) ** 2
        img_token_id = 32000
        labels = []
        for token_id in input_ids[0].tolist():
            if token_id == img_token_id:
                labels.extend([f"<IMG{i + 1:03d}>" for i in range(num_patches)])
            else:
                labels.append(tokenizer.decode([token_id]))
        return labels

    def _build_image_token_mask(
        self, input_ids: torch.Tensor, image_size: int, patch_size: int
    ) -> List[bool]:
        num_patches = (image_size // patch_size) ** 2
        img_token_id = 32000
        mask = []
        for token_id in input_ids[0].tolist():
            if token_id == img_token_id:
                mask.extend([True] * num_patches)
            else:
                mask.append(False)
        return mask

    def get_token_predictions(
        self, result: LogitLensResult, layer_idx: int, token_idx: int
    ) -> List[TokenPrediction]:
        if layer_idx < 0 or layer_idx >= len(result.layer_predictions):
            return []
        layer = result.layer_predictions[layer_idx]
        if token_idx < 0 or token_idx >= len(layer.predictions):
            return []
        return layer.predictions[token_idx]

    def get_convergence_point(
        self, result: LogitLensResult, token_idx: int, target_token: str
    ) -> Optional[int]:
        for layer in result.layer_predictions:
            if token_idx < len(layer.predictions):
                top_pred = layer.predictions[token_idx][0]
                if top_pred.token.lower() == target_token.lower():
                    return layer.layer_index
        return None

    def get_patch_confidences(
        self, result: LogitLensResult, layer_idx: int
    ) -> List[Tuple[int, float]]:
        confidences = []
        if layer_idx < 0 or layer_idx >= len(result.layer_predictions):
            return confidences

        layer = result.layer_predictions[layer_idx]
        for idx, (is_image, predictions) in enumerate(
            zip(result.image_token_mask, layer.predictions)
        ):
            if is_image and predictions:
                confidences.append((idx, predictions[0].probability))

        return sorted(confidences, key=lambda x: x[1], reverse=True)
