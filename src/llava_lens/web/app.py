import base64
import json
from io import BytesIO
from pathlib import Path

import spaces
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from PIL import Image

app = FastAPI(title="LLAVA-LENS", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = Path(__file__).parent
static_dir = base_dir / "static"
templates_dir = base_dir / "templates"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

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

    print("Loading model...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("Model loaded")


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


def build_dashboard_html(image_b64: str, result: dict, prompt: str) -> str:
    token_labels = result["token_labels"]
    image_mask = result["image_mask"]
    layer_data = result["layer_data"]
    num_layers = result["num_layers"]
    gs = IMAGE_SIZE // PATCH_SIZE

    text_rows = ""
    img_rows = ""

    for t, label in enumerate(token_labels):
        cells = ""
        for l in range(num_layers):
            preds = layer_data[l][t]
            top = preds[0]
            p = top["p"]
            alpha = min(0.6, p * 0.6)
            color = "#fff" if p > 0.8 else "#94a3b8" if p > 0.3 else "#475569"
            cells += f'<td style="background:rgba(99,102,241,{alpha:.2f});color:{color};font-size:11px;padding:4px 6px;" title="L{l+1}: {top["t"]} ({p*100:.1f}%)">{top["t"]}</td>'

        if image_mask[t]:
            img_rows += f'<tr class="patch-row" data-label="{label}"><td style="padding:4px 8px;"><span style="background:rgba(239,68,68,0.15);color:#f87171;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;">{label}</span></td>{cells}</tr>'
        else:
            text_rows += f'<tr><td style="padding:4px 8px;font-weight:500;">{label}</td>{cells}</tr>'

    layer_headers = "".join([f'<th style="padding:8px;font-size:11px;">L{i+1}</th>' for i in range(num_layers)])

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Inter',system-ui,sans-serif;background:#09090b;color:#fafafa;}}
.wrap{{display:flex;height:100vh;}}
.side{{width:380px;background:#111113;border-right:1px solid rgba(255,255,255,0.06);display:flex;flex-direction:column;overflow-y:auto;}}
.logo{{padding:24px;border-bottom:1px solid rgba(255,255,255,0.06);}}
.logo h1{{font-size:20px;font-weight:700;background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.logo .sub{{font-size:12px;color:#52525b;margin-top:4px;font-family:monospace;}}
.img-area{{padding:20px;flex:1;display:flex;flex-direction:column;align-items:center;overflow-y:auto;}}
.img-frame{{position:relative;width:336px;height:336px;border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);box-shadow:0 25px 50px rgba(0,0,0,0.5);}}
.img-frame img{{width:100%;height:100%;object-fit:cover;}}
.patch-hl{{position:absolute;border:2px solid #ef4444;background:rgba(239,68,68,0.2);pointer-events:none;display:none;box-shadow:0 0 20px rgba(239,68,68,0.5);border-radius:2px;z-index:10;}}
.info{{margin-top:16px;padding:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;width:100%;font-size:13px;color:#a1a1aa;}}
.info strong{{color:#e2e8f0;}}
.main{{flex:1;display:flex;flex-direction:column;overflow:hidden;}}
.tbl-hdr{{padding:16px 24px;border-bottom:1px solid rgba(255,255,255,0.06);display:flex;align-items:center;justify-content:space-between;background:#111113;}}
.tbl-hdr h2{{font-size:14px;font-weight:600;}}
.tabs{{display:flex;gap:4px;}}
.tab{{padding:6px 14px;border-radius:6px;font-size:12px;font-weight:500;cursor:pointer;background:transparent;color:#71717a;border:1px solid transparent;transition:all 0.2s;}}
.tab:hover{{color:#fafafa;background:rgba(255,255,255,0.05);}}
.tab.active{{background:rgba(99,102,241,0.15);color:#818cf8;border-color:rgba(99,102,241,0.3);}}
.tbl-wrap{{flex:1;overflow:auto;}}
table{{border-collapse:collapse;width:100%;font-size:12px;}}
th,td{{padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.03);border-right:1px solid rgba(255,255,255,0.03);white-space:nowrap;text-align:left;}}
thead th{{position:sticky;top:0;background:#09090b;z-index:10;font-weight:600;color:#52525b;font-size:11px;padding:8px 10px;border-bottom:2px solid rgba(255,255,255,0.08);}}
.rh{{position:sticky;left:0;background:#09090b;z-index:5;text-align:right;min-width:110px;border-right:2px solid rgba(255,255,255,0.08);color:#d4d4d8;font-weight:500;}}
tr:hover td{{background:rgba(99,102,241,0.05);}}
td:hover{{outline:1px solid #818cf8;outline-offset:-1px;z-index:1;position:relative;}}
.ttip{{position:fixed;background:#18181b;color:white;padding:14px;border-radius:10px;font-size:12px;pointer-events:none;z-index:1000;display:none;border:1px solid rgba(255,255,255,0.1);box-shadow:0 25px 50px rgba(0,0,0,0.8);min-width:220px;}}
.ttip-h{{font-weight:600;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.08);color:#818cf8;}}
.ttip-r{{display:flex;justify-content:space-between;margin-bottom:4px;}}
.ttip-r span:last-child{{color:#52525b;}}
.ttip-b{{height:3px;background:rgba(255,255,255,0.06);border-radius:2px;margin-bottom:8px;}}
.ttip-f{{height:100%;background:linear-gradient(90deg,#6366f1,#818cf8);border-radius:2px;}}
</style></head><body>
<div class="wrap">
<div class="side">
<div class="logo"><h1>LLAVA-LENS</h1><div class="sub">llava-1.5-7b-hf</div></div>
<div class="img-area">
<div class="img-frame"><img src="data:image/png;base64,{image_b64}"><div class="patch-hl" id="phl"></div></div>
<div class="info"><strong>Prompt:</strong> {prompt}</div>
<div style="margin-top:12px;padding:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;width:100%;font-size:12px;color:#71717a;">
<div>Hover cells for predictions</div>
<div>Blue intensity = confidence</div>
<div>IMG rows = image patches</div>
</div>
</div>
</div>
<div class="main">
<div class="tbl-hdr"><h2>Token Predictions Across Layers</h2>
<div class="tabs"><div class="tab active" onclick="showTab('all')">All</div><div class="tab" onclick="showTab('text')">Text</div><div class="tab" onclick="showTab('image')">Image</div></div></div>
<div class="tbl-wrap"><table><thead><tr><th class="rh">Token</th>{layer_headers}</tr></thead><tbody id="tb">
{text_rows}
<tr id="sep" style="display:none;"><td colspan="{num_layers+1}" style="padding:4px;background:rgba(239,68,68,0.05);border-bottom:1px solid rgba(239,68,68,0.2);"><span style="font-size:10px;color:#f87171;text-transform:uppercase;letter-spacing:1px;">Image Patches</span></td></tr>
{img_rows}
</tbody></table></div>
</div>
</div>
<div class="ttip" id="ttip"></div>
<script>
const GS={gs},PS={PATCH_SIZE};
document.querySelectorAll('td').forEach(td=>{{
td.addEventListener('mouseenter',function(){{
const t=this.getAttribute('title');if(!t)return;
const p=t.split(': ');const l=p[0];const r=p[1]?p[1].split(' ('):['',''];const tk=r[0];const pr=r[1]?r[1].replace(')',''):'';
let h=`<div class="ttip-h">${{l}}</div><div class="ttip-r"><span>${{tk}}</span><span>${{pr}}</span></div>`;
const pct=parseFloat(pr)||0;h+=`<div class="ttip-b"><div class="ttip-f" style="width:${{pct}}%"></div></div>`;
const tt=document.getElementById('ttip');tt.innerHTML=h;tt.style.display='block';
const rc=this.getBoundingClientRect();let left=rc.right+12,top=rc.top;
if(left+240>window.innerWidth)left=rc.left-252;if(top+150>window.innerHeight)top=window.innerHeight-150;
tt.style.left=left+'px';tt.style.top=top+'px';this.style.outline='1px solid #818cf8';
}});
td.addEventListener('mouseleave',function(){{document.getElementById('ttip').style.display='none';this.style.outline='';}});
}});
document.querySelectorAll('.patch-row').forEach(tr=>{{
tr.addEventListener('mouseenter',function(){{
const l=this.dataset.label;const idx=parseInt(l.replace('<IMG','').replace('>',''))-1;
const r=Math.floor(idx/GS),c=idx%GS;
const hl=document.getElementById('phl');
hl.style.width=PS+'px';hl.style.height=PS+'px';hl.style.left=(c*PS)+'px';hl.style.top=(r*PS)+'px';hl.style.display='block';
}});
tr.addEventListener('mouseleave',()=>document.getElementById('phl').style.display='none');
}});
function showTab(t){{document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));event.target.classList.add('active');
const sep=document.getElementById('sep');const pr=document.querySelectorAll('.patch-row');
const tr=document.querySelectorAll('#tb tr:not(.patch-row):not(#sep)');
if(t==='all'){{sep.style.display='none';tr.forEach(r=>r.style.display='');pr.forEach(r=>r.style.display='');}}
else if(t==='text'){{sep.style.display='none';tr.forEach(r=>r.style.display='');pr.forEach(r=>r.style.display='none');}}
else if(t==='image'){{sep.style.display='';tr.forEach(r=>r.style.display='none');pr.forEach(r=>r.style.display='');}}
}}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request})


@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    return templates.TemplateResponse("batch.html", {"request": request})


@app.get("/runs", response_class=HTMLResponse)
async def runs_page(request: Request):
    return templates.TemplateResponse("runs.html", {"request": request})


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    return templates.TemplateResponse("models.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})


@app.post("/api/analyze")
async def analyze_api(file: UploadFile = File(...), prompt: str = "Describe the image."):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        result = analyze_image_gpu(image, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    buffered = BytesIO()
    image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return HTMLResponse(content=build_dashboard_html(img_b64, result, prompt))


@app.post("/api/analyze/json")
async def analyze_json(file: UploadFile = File(...), prompt: str = "Describe the image."):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        result = analyze_image_gpu(image, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return JSONResponse({
        "status": "success",
        "model": MODEL_NAME,
        "prompt": prompt,
        "num_layers": result["num_layers"],
        "token_labels": result["token_labels"],
        "image_mask": result["image_mask"],
        "layer_data": result["layer_data"],
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "model_loaded": model is not None}


@app.get("/api/models")
async def list_models():
    from llava_lens.core.registry import get_registry
    registry = get_registry()
    return {"models": registry.list_available("models")}


@app.get("/api/runs")
async def list_runs(limit: int = 50):
    from llava_lens.tracking.tracker import ExperimentTracker
    from llava_lens.core.config import get_config
    tracker = ExperimentTracker(get_config())
    runs = tracker.list_runs(limit)
    return {"runs": [run.to_dict() for run in runs]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    from llava_lens.tracking.tracker import ExperimentTracker
    from llava_lens.core.config import get_config
    tracker = ExperimentTracker(get_config())
    run = tracker.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.to_dict()


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    from llava_lens.tracking.tracker import ExperimentTracker
    from llava_lens.core.config import get_config
    tracker = ExperimentTracker(get_config())
    success = tracker.delete_run(run_id)
    if not success:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "deleted"}
