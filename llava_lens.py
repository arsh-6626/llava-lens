import torch
import json
from pathlib import Path
import base64
import os
from io import BytesIO
from PIL import Image


def create_logit_lens(hidden_states, norm, lm_head, tokenizer, image, model_name, image_filename, prompt, save_folder = ".", image_size=336, patch_size=14, misc_text=""):
    input_ids = tokenizer.encode(prompt)
    img_token_id = 32000  # The token ID for <img>
    img_token_count = (image_size // patch_size) ** 2  # 576 for 336x336 image with 14x14 patches
    
    token_labels = []
    for token_id in input_ids:
        if token_id == img_token_id:
            token_labels.extend([f"<IMG{(i+1):03d}>" for i in range(img_token_count)])
        else:
            token_labels.append(tokenizer.decode([token_id]))
        num_layers = len(hidden_states)
    sequence_length = hidden_states[0].size(1)
    
    all_top_tokens = []
    
    for layer in range(num_layers):
        layer_hidden_states = hidden_states[layer]
        normalized = norm(layer_hidden_states)
        logits = lm_head(normalized)
        probs = torch.softmax(logits, dim=-1)
        top_5_values, top_5_indices = torch.topk(probs, k=5, dim=-1)
        layer_top_tokens = []
        for pos in range(sequence_length):
            top_5_tokens = [tokenizer.decode(idx.item()) for idx in top_5_indices[0, pos]]
            top_5_probs = [f"{prob.item():.4f}" for prob in top_5_values[0, pos]]
            layer_top_tokens.append(list(zip(top_5_tokens, top_5_probs)))
        
        all_top_tokens.append(layer_top_tokens)
    img_w, img_h = image.size
    min_dim = min(img_w, img_h)
    left = (img_w - min_dim) / 2
    top = (img_h - min_dim) / 2
    right = (img_w + min_dim) / 2
    bottom = (img_h + min_dim) / 2
    image_cropped = image.crop((left, top, right, bottom))
    image_resized = image_cropped.resize((image_size, image_size), Image.LANCZOS)
    buffered = BytesIO()
    image_resized.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    template_path = "./result.html"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at: {template_path}")
        
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace('DATA_PLACEHOLDER', json.dumps(all_layer_data))
    html_content = html_content.replace('LABELS_PLACEHOLDER', json.dumps(token_labels))
    html_content = html_content.replace('MASK_PLACEHOLDER', json.dumps(is_image_token_mask))
    html_content = html_content.replace('IMAGE_PLACEHOLDER', img_str)
    html_content = html_content.replace('MODELNAME_PLACEHOLDER', model_name)
    html_content = html_content.replace('PROMPT_PLACEHOLDER', prompt)
    os.makedirs(save_folder, exist_ok=True)
    output_filename = f"{model_name}_{Path(image_filename).stem}_dashboard.html"
    output_path = Path(save_folder) / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to: {output_path}")