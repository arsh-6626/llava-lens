import base64
import json
from io import BytesIO

import spaces
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image

app = FastAPI(title="LLAVA-LENS", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

model = None
processor = None

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
IMAGE_TOKEN_ID = 32000
IMAGE_SIZE = 336
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2


def load_model():
    global model, processor
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)


@spaces.GPU(duration=120)
def analyze_image_gpu(image: Image.Image, prompt: str):
    global model, processor
    if model is None:
        load_model()

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )

    hidden_states = outputs.hidden_states
    input_ids = inputs["input_ids"]
    norm = model.language_model.model.norm
    lm_head = model.language_model.lm_head

    token_labels = []
    for token_id in input_ids[0].tolist():
        if token_id == IMAGE_TOKEN_ID:
            token_labels.extend([f"<IMG{i + 1:03d}>" for i in range(NUM_PATCHES)])
        else:
            token_labels.append(processor.tokenizer.decode([token_id]))

    image_mask = []
    for token_id in input_ids[0].tolist():
        if token_id == IMAGE_TOKEN_ID:
            image_mask.extend([True] * NUM_PATCHES)
        else:
            image_mask.append(False)

    num_layers = len(hidden_states)
    all_layer_data = []

    for layer_idx in range(num_layers):
        layer_hidden = hidden_states[layer_idx]
        normalized = norm(layer_hidden)
        logits = lm_head(normalized)
        probs = torch.softmax(logits, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)

        layer_data = []
        for pos in range(top5_indices.shape[1]):
            preds = []
            for k in range(5):
                token_text = processor.tokenizer.decode([top5_indices[0, pos, k].item()])
                prob = top5_probs[0, pos, k].item()
                preds.append({"t": token_text, "p": round(prob, 4)})
            layer_data.append(preds)
        all_layer_data.append(layer_data)

    return {
        "token_labels": token_labels,
        "image_mask": image_mask,
        "layer_data": all_layer_data,
        "num_layers": num_layers,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = static_dir / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), prompt: str = "Describe the image."):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = analyze_image_gpu(image, prompt)

    buffered = BytesIO()
    image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse({
        "image": img_b64,
        "prompt": prompt,
        "model": MODEL_NAME,
        "token_labels": result["token_labels"],
        "image_mask": result["image_mask"],
        "layer_data": result["layer_data"],
        "num_layers": result["num_layers"],
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}
