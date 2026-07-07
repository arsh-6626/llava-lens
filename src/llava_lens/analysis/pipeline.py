from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger
from llava_lens.models.base import BaseModelWrapper, ModelOutput
from llava_lens.analysis.logit_lens import LogitLensAnalyzer, LogitLensResult
from llava_lens.analysis.attention import AttentionAnalyzer, AttentionResult
from llava_lens.analysis.patch_selection import PatchSelector, PatchSelectionResult

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    model_output: ModelOutput
    logit_lens: LogitLensResult
    attention: AttentionResult
    patch_selection: PatchSelectionResult
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalysisPipeline:
    def __init__(self, config: Config, model: BaseModelWrapper):
        self.config = config
        self.model = model
        self.logit_lens = LogitLensAnalyzer(config)
        self.attention = AttentionAnalyzer(config)
        self.patch_selector = PatchSelector(config)

    def analyze(
        self,
        image: Any,
        prompt: str,
        layer_idx: int = -1,
    ) -> AnalysisResult:
        logger.info(f"Starting analysis for prompt: {prompt[:50]}...")

        self.model.ensure_loaded()

        model_output = self.model.forward(
            image=image,
            prompt=prompt,
            output_hidden_states=True,
            output_attentions=True,
        )

        logit_lens_result = self.logit_lens.analyze(
            hidden_states=model_output.hidden_states,
            norm_layer=self.model.get_norm_layer(),
            lm_head=self.model.get_lm_head(),
            tokenizer=self.model.get_tokenizer(),
            input_ids=model_output.input_ids,
            image_size=self.model.get_image_size(),
            patch_size=self.model.get_patch_size(),
        )

        attention_result = self.attention.analyze(
            attentions=model_output.attentions,
            image_token_indices=model_output.image_token_indices,
        )

        patch_selection_result = self.patch_selector.select(
            logit_lens_result=logit_lens_result,
            layer_idx=layer_idx,
        )

        return AnalysisResult(
            model_output=model_output,
            logit_lens=logit_lens_result,
            attention=attention_result,
            patch_selection=patch_selection_result,
            metadata={
                "prompt": prompt,
                "model_name": self.config.model.name,
                "layer_idx": layer_idx,
            },
        )

    def analyze_batch(
        self,
        images: List[Any],
        prompts: List[str],
        layer_idx: int = -1,
    ) -> List[AnalysisResult]:
        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")

        results = []
        for image, prompt in zip(images, prompts):
            try:
                result = self.analyze(image, prompt, layer_idx)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze image: {e}")
                results.append(None)

        return results
