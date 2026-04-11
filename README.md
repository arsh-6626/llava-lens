## Overview
This project focuses on improving attention to fine-grained details in Vision-Language Models (VLMs) using the Logit Lens. Instead of relying only on final predictions, it inspects intermediate transformer representations to understand how semantic concepts like blood or injury emerge across layers. The goal is to improve classification in tasks where subtle visual cues matter more than dominant objects, such as trauma detection.

## Patch Types
- High-confidence patches (>1%)  
  These contain meaningful information and capture important visual cues such as wounds or blood. They are used for downstream classification.

- Low-confidence patches (<1%)  
  These do not carry strong semantic meaning and act as soft prompts that help guide representations but are not directly useful for decisions.

## Pipeline
```
Image → VLM → Hidden States → Attention Filtering  
→ Logit Lens → Patch Selection  
→ Transformer Classifier → Prediction  
```

Steps:
1. Extract intermediate hidden states from a pretrained VLM  
2. Use attention maps to identify important regions  
3. Apply logit lens to evaluate semantic strength of patches  
4. Select high-confidence patches with some contextual neighbors  
5. Feed filtered embeddings into a lightweight classifier  
6. Output trauma / no-trauma prediction with interpretability  

## Repository Structure
```
├── README.md  
├── main.py  
├── llava_lens.py  
├── hooked_llava.py  
├── get_embeds.py  
├── result.html  
```
## Author
Arsh Abbas Naqvi 
Delhi Technological University
