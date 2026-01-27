import torch
from transformers import PreTrainedTokenizerBase

class InputsEmbeds:
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase, 
                 activation: torch.Tensor, 
                 prompt: str,
                 model_id=""):
        self.tokenizer = tokenizer
        self.activation = activation
        self.prompt = prompt
        if model_id == "xtuner/llava-phi-3-mini-hf":
            self.img_token_id = 32038
        elif model_id == "llava-hf/llava-1.5-7b-hf":
            self.img_token_id = 32000
        else:
            self.img_token_id = 32000
            print(f"WARNING: model_id provided '{model_id}' not in list, defaulting to LLaVA 1.5")
        self.img_token_count = 576 

    def get_img_and_text_embed(self):
        input_ids = torch.tensor(self.tokenizer.encode(self.prompt))
        is_img_token = input_ids == self.img_token_id
        img_token_indices = torch.where(is_img_token)[0]
        is_img_token_mask = torch.zeros(self.activation.shape[1], dtype=torch.bool)
        img_start_end_indices = []
        for i, idx in enumerate(img_token_indices): 
               start = idx + i * self.img_token_count
               end = start + self.img_token_count
               is_img_token_mask[start:end] = True
               img_start_end_indices.append((start, end))
        img_embeds = self.activation[:, is_img_token_mask, :]
        text_embeds = self.activation[:, ~is_img_token_mask, :]

        return img_embeds, text_embeds, img_start_end_indices
    