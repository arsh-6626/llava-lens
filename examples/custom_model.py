from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
from PIL import Image

from llava_lens.core.config import Config
from llava_lens.core.registry import get_registry
from llava_lens.models.base import BaseModelWrapper, ModelOutput


@get_registry().register_model("custom-vlm")
class CustomVLMWrapper(BaseModelWrapper):
    def __init__(self, config: Config):
        super().__init__(config)
        self._image_token_id = 32000
        self._image_size = config.model.image_size
        self._patch_size = config.model.patch_size

    def load(self) -> None:
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            device_map=self.config.model.device,
            torch_dtype=getattr(torch, self.config.model.torch_dtype, torch.float16),
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model.name)
        self._loaded = True

    def forward(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
    ) -> ModelOutput:
        self.ensure_loaded()
        pil_image = self.load_image(image)

        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

        return ModelOutput(
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            logits=outputs.logits,
            input_ids=inputs.get("input_ids"),
        )

    def generate(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
    ) -> str:
        self.ensure_loaded()
        pil_image = self.load_image(image)

        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def get_tokenizer(self) -> Any:
        self.ensure_loaded()
        return self.processor.tokenizer

    def get_norm_layer(self) -> Any:
        self.ensure_loaded()
        return self.model.model.norm

    def get_lm_head(self) -> Any:
        self.ensure_loaded()
        return self.model.lm_head

    def get_image_token_id(self) -> int:
        return self._image_token_id

    def get_image_size(self) -> int:
        return self._image_size

    def get_patch_size(self) -> int:
        return self._patch_size
