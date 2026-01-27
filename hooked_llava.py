from ast import mod
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
from contextlib import contextmanager
from typing import Callable, Union, Dict, Any
import os

@contextmanager
def session_hook(model, hook):
    handle = model.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()

class HookedLlava:
    def __init__(self, model_name, hook_loc):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.hook_loc = hook_loc
        self.data = None

    def ablate_inputs(self, indices, replacement_tensor):
 
        def ablation_hook(module, args, kwargs):
            input_embeds = kwargs.get('inputs_embeds', None)
            if input_embeds.shape[-2]==1:
                return args, kwargs
            modified_embeds = input_embeds.clone()
            local_replacement_tensor = replacement_tensor.to(modified_embeds.dtype).to(modified_embeds.device)
            modified_embeds[:, indices, :] = local_replacement_tensor
            kwargs['inputs_embeds'] = modified_embeds
            return args, kwargs
    
        hook = self.model.language_model.register_forward_pre_hook(ablation_hook, with_kwargs=True)
        try:
            yield
        finally:
            hook.remove()

    def prompt_hook(self, module, args, kwargs, output):
        self.data = kwargs['inputs_embeds']
        return output
    
    def forward(self, 
                image_path_or_image: Union[str, Image.Image], 
                prompt: str,
                output_hidden_states=False,
                output_attentions=False,
                ):
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image)
        else:
            image = image_path_or_image
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        return outputs
        # if output_hidden_states:
        #     return outputs.hidden_states

    def generate(self, image_path, prompt, max_new_tokens, output_hidden_states, do_sample):
        image = Image.open(image_path)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs.to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states, do_sample=do_sample)
        
        response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        if output_hidden_states:
            return response, output.hidden_states
        
        return response
    
    def get_text_model_in(self, image_path_or_image: Union[str, Image.Image], prompt: str,):
        
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image)
        else:
            image = image_path_or_image
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs.to(self.model.device)
        if self.hook_loc == "text_model_in":
            with session_hook(self.model.language_model, self.prompt_hook):
                with torch.no_grad():
                    outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
        

        return self.data