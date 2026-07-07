from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionResult:
    attention_maps: List[torch.Tensor]
    aggregated_attention: torch.Tensor
    num_heads: int
    num_layers: int
    head_importance: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttentionAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.head_agg = config.analysis.attention.head_aggregation
        self.layer_agg = config.analysis.attention.layer_aggregation

    def analyze(
        self,
        attentions: Tuple[torch.Tensor],
        image_token_indices: Optional[torch.Tensor] = None,
    ) -> AttentionResult:
        if attentions is None or len(attentions) == 0:
            raise ValueError("No attention maps provided")

        attention_maps = [att.squeeze(0) for att in attentions]
        num_layers = len(attention_maps)
        num_heads = attention_maps[0].shape[0]

        head_aggregated = self._aggregate_heads(attention_maps)
        layer_aggregated = self._aggregate_layers(head_aggregated)

        head_importance = self._compute_head_importance(attention_maps)

        return AttentionResult(
            attention_maps=attention_maps,
            aggregated_attention=layer_aggregated,
            num_heads=num_heads,
            num_layers=num_layers,
            head_importance=head_importance,
        )

    def _aggregate_heads(self, attention_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        aggregated = []
        for att in attention_maps:
            if self.head_agg == "mean":
                agg = att.mean(dim=0)
            elif self.head_agg == "max":
                agg = att.max(dim=0).values
            elif self.head_agg == "sum":
                agg = att.sum(dim=0)
            else:
                agg = att.mean(dim=0)
            aggregated.append(agg)
        return aggregated

    def _aggregate_layers(self, head_aggregated: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(head_aggregated)
        if self.layer_agg == "last":
            return stacked[-1]
        elif self.layer_agg == "mean":
            return stacked.mean(dim=0)
        elif self.layer_agg == "sum":
            return stacked.sum(dim=0)
        return stacked[-1]

    def _compute_head_importance(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        importance = []
        for att in attention_maps:
            head_imp = att.mean(dim=(1, 2))
            importance.append(head_imp)
        return torch.stack(importance).mean(dim=0)

    def get_image_attention(
        self, result: AttentionResult, image_token_indices: torch.Tensor
    ) -> torch.Tensor:
        if len(image_token_indices) == 0:
            return torch.tensor([])
        att = result.aggregated_attention
        image_att = att[:, image_token_indices]
        return image_att.mean(dim=-1)

    def get_top_attended_patches(
        self, result: AttentionResult, image_token_indices: torch.Tensor, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        image_att = self.get_image_attention(result, image_token_indices)
        if len(image_att) == 0:
            return []

        values, indices = torch.topk(image_att, k=min(top_k, len(image_att)))
        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]
