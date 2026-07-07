import base64
import json
from io import BytesIO
from pathlib import Path

import spaces
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI(title="LLAVA-LENS", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
processor = None
pipeline = None

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
IMAGE_TOKEN_ID = 32000
IMAGE_SIZE = 336
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2


def load_model():
    global model, processor, pipeline
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
    attentions = outputs.attentions
    input_ids = inputs["input_ids"]

    norm = model.language_model.model.norm
    lm_head = model.language_model.lm_head

    token_labels = []
    img_count = 0
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
                preds.append({"t": token_text, "p": prob, "d": [(token_text, prob)]})
            layer_data.append(preds[0] if preds else {"t": "", "p": 0, "d": []})
        all_layer_data.append(layer_data)

    return {
        "token_labels": token_labels,
        "image_mask": image_mask,
        "layer_data": all_layer_data,
        "num_layers": num_layers,
    }


def generate_dashboard_html(image_b64: str, result: dict, prompt: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLAVA-LENS Analysis</title>
    <style>
        :root {{ --bg: #0f172a; --sidebar: #1e293b; --text: #f1f5f9; --muted: #94a3b8; --border: #334155; --primary: #60a5fa; }}
        body {{ margin: 0; font-family: system-ui; background: var(--bg); color: var(--text); display: flex; height: 100vh; }}
        .sidebar {{ width: 400px; background: var(--sidebar); border-right: 1px solid var(--border); display: flex; flex-direction: column; }}
        .header {{ padding: 20px; border-bottom: 1px solid var(--border); }}
        .header h1 {{ margin: 0; font-size: 20px; }}
        .img-wrap {{ padding: 20px; flex: 1; display: flex; flex-direction: column; align-items: center; overflow-y: auto; }}
        .img-container {{ position: relative; width: 336px; height: 336px; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }}
        .highlight {{ position: absolute; border: 2px solid #ef4444; background: rgba(239,68,68,0.25); display: none; pointer-events: none; }}
        .prompt {{ margin-top: 15px; padding: 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; font-size: 12px; color: var(--muted); width: 100%; box-sizing: border-box; }}
        .main {{ flex: 1; overflow: auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; font-size: 12px; border: 1px solid var(--border); white-space: nowrap; }}
        th {{ background: var(--sidebar); position: sticky; top: 0; z-index: 10; }}
        .row-header {{ position: sticky; left: 0; background: var(--sidebar); z-index: 5; text-align: right; min-width: 120px; }}
        .img-row {{ display: none; }}
        .conf-high {{ color: #fff; font-weight: 600; }}
        .conf-med {{ color: #94a3b8; }}
        .conf-low {{ color: #64748b; }}
        #tooltip {{ position: fixed; background: rgba(15,23,42,0.95); color: white; padding: 12px; border-radius: 8px; font-size: 12px; pointer-events: none; z-index: 100; display: none; border: 1px solid rgba(255,255,255,0.15); min-width: 200px; }}
        .tt-row {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
        .tt-bar {{ height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-bottom: 8px; }}
        .tt-fill {{ height: 100%; background: var(--primary); border-radius: 2px; }}
        .controls {{ padding: 15px; border-bottom: 1px solid var(--border); display: flex; gap: 10px; flex-direction: column; }}
        .toggle {{ display: flex; justify-content: space-between; align-items: center; font-size: 13px; color: var(--muted); }}
        .switch {{ width: 36px; height: 20px; background: #475569; border-radius: 10px; position: relative; cursor: pointer; }}
        .switch.on {{ background: var(--primary); }}
        .switch::after {{ content: ''; position: absolute; width: 16px; height: 16px; background: white; border-radius: 50%; top: 2px; left: 2px; transition: 0.2s; }}
        .switch.on::after {{ left: 18px; }}
    </style>
</head>
<body>
<div class="sidebar">
    <div class="header">
        <h1>LLAVA-LENS</h1>
        <div style="font-size:12px;color:var(--muted);font-family:monospace;">Model: llava-1.5-7b</div>
    </div>
    <div class="controls">
        <div class="toggle"><span>Show Image Tokens</span><div class="switch" id="toggleImg" onclick="toggleImg()"></div></div>
        <div class="toggle"><span>Confidence Heatmap</span><div class="switch on" id="toggleHeat" onclick="toggleHeat()"></div></div>
    </div>
    <div class="img-wrap">
        <div class="img-container">
            <img src="data:image/png;base64,{image_b64}" width="336" height="336">
            <div class="highlight"></div>
        </div>
        <div class="prompt"><strong style="color:white">Prompt:</strong><br>{prompt}</div>
    </div>
</div>
<div class="main">
    <table id="tbl"><thead><tr id="hdr"><th class="row-header">Token</th></tr></thead><tbody id="body"></tbody></table>
</div>
<div id="tooltip"></div>
<script>
const LD = {json.dumps(result['layer_data'])};
const TL = {json.dumps(result['token_labels'])};
const IM = {json.dumps(result['image_mask'])};
const GS = {IMAGE_SIZE // PATCH_SIZE};
const NL = LD.length;
const body = document.getElementById('body');
const hdr = document.getElementById('hdr');
const tt = document.getElementById('tooltip');
const hl = document.querySelector('.highlight');
const imgC = document.querySelector('.img-container');

function init() {{
    for(let i=0;i<NL;i++){{const th=document.createElement('th');th.textContent=i+1;hdr.appendChild(th);}}
    for(let t=0;t<TL.length;t++){{
        const tr=document.createElement('tr');tr.id='r-'+t;
        const th=document.createElement('th');th.className='row-header';th.textContent=TL[t];tr.appendChild(th);
        if(IM[t]){{tr.classList.add('img-row');tr.style.display='none';}}
        for(let l=0;l<NL;l++){{
            const td=document.createElement('td');
            const c=LD[l][t];td.textContent=c.t;td.dataset.prob=c.p;td.dataset.token=c.t.toLowerCase();
            td.onmouseenter=(e)=>showTT(e,l,t);td.onmouseleave=hideTT;tr.appendChild(td);
        }}
        tr.onmouseenter=()=>hlPatch(t);tr.onmouseleave=hideHL;body.appendChild(tr);
    }}
    updateHeat();
}}

function toggleImg(){{
    const sw=document.getElementById('toggleImg');sw.classList.toggle('on');
    document.querySelectorAll('.img-row').forEach(r=>r.style.display=sw.classList.contains('on')?'table-row':'none');
}}
function toggleHeat(){{
    const sw=document.getElementById('toggleHeat');sw.classList.toggle('on');updateHeat();
}}
function updateHeat(){{
    const on=document.getElementById('toggleHeat').classList.contains('on');
    document.querySelectorAll('td').forEach(td=>{{
        if(!on){{td.style.backgroundColor='';td.className='';return;}}
        const p=parseFloat(td.dataset.prob);
        td.style.backgroundColor='rgba(59,130,246,'+Math.max(0,p*0.4)+')';
        if(p>0.8)td.classList.add('conf-high');else if(p>0.3)td.classList.add('conf-med');else td.classList.add('conf-low');
    }});
}}
function showTT(e,l,t){{
    const c=LD[l][t];const r=e.target.getBoundingClientRect();
    let h='<div style="font-weight:bold;margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:6px;color:var(--primary)">L'+(l+1)+' | '+TL[t]+'</div>';
    c.d.forEach(i=>{{const p=(i[1]*100).toFixed(1);h+='<div class="tt-row"><span>'+i[0]+'</span><span style="color:var(--muted)">'+p+'%</span></div><div class="tt-bar"><div class="tt-fill" style="width:'+p+'%"></div></div>';}});
    tt.innerHTML=h;tt.style.display='block';
    let left=r.right+15,top=r.top;
    if(left+220>window.innerWidth)left=r.left-235;
    tt.style.left=left+'px';tt.style.top=top+'px';
    e.target.style.outline='2px solid var(--primary)';
}}
function hideTT(e){{tt.style.display='none';if(e.target)e.target.style.outline='';}}
function hlPatch(t){{
    const l=TL[t];if(!l||!l.startsWith('<IMG')){{hideHL();return;}}
    const p=parseInt(l.replace('<IMG','').replace('>',''))-1;
    hl.style.width='{PATCH_SIZE}px';hl.style.height='{PATCH_SIZE}px';
    hl.style.left=(p%GS*{PATCH_SIZE})+'px';hl.style.top=(Math.floor(p/GS)*{PATCH_SIZE})+'px';
    hl.style.display='block';
}}
function hideHL(){{hl.style.display='none';}}
imgC.onmousemove=(e)=>{{
    if(!document.getElementById('toggleImg').classList.contains('on'))return;
    const r=imgC.getBoundingClientRect();
    const c=Math.floor((e.clientX-r.left)/{PATCH_SIZE}),rw=Math.floor((e.clientY-r.top)/{PATCH_SIZE});
    if(c>={IMAGE_SIZE//PATCH_SIZE}||rw>={IMAGE_SIZE//PATCH_SIZE})return;
    const i=rw*{IMAGE_SIZE//PATCH_SIZE}+c+1;
    const ti=TL.indexOf('<IMG'+i.toString().padStart(3,'0')+'>');if(ti!==-1)hlPatch(ti);
}};
imgC.onmouseleave=hideHL;
init();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html><head><title>LLAVA-LENS</title>
<style>
body{margin:0;font-family:system-ui;background:#0f172a;color:#f1f5f9;min-height:100vh;display:flex;align-items:center;justify-content:center;flex-direction:column;}
.card{background:#1e293b;border:2px dashed #334155;border-radius:12px;padding:60px;text-align:center;max-width:600px;width:90%;}
.card:hover{border-color:#60a5fa;}
h1{margin-bottom:10px;}
p{color:#94a3b8;margin-bottom:20px;}
.btn{padding:12px 24px;background:#60a5fa;color:#0f172a;border:none;border-radius:8px;font-size:16px;cursor:pointer;font-weight:600;}
.btn:hover{background:#93bbfd;}
.btn:disabled{background:#475569;cursor:not-allowed;}
input[type=file]{display:none;}
.preview{margin-top:20px;display:none;}
.preview img{max-width:100%;border-radius:8px;border:1px solid #334155;}
#loading{display:none;margin-top:20px;}
.spinner{border:3px solid #334155;border-top:3px solid #60a5fa;border-radius:50%;width:30px;height:30px;animation:spin 1s linear infinite;margin:0 auto 10px;}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
</style></head>
<body>
<div class="card">
<h1>LLAVA-LENS</h1>
<p>Upload an image for Logit Lens analysis</p>
<input type="file" id="fileInput" accept="image/*">
<button class="btn" onclick="document.getElementById('fileInput').click()">Choose Image</button>
<div class="preview" id="preview"><img id="previewImg"></div>
<div id="loading"><div class="spinner"></div>Analyzing...</div>
</div>
<script>
document.getElementById('fileInput').onchange=async(e)=>{
const file=e.target.files[0];if(!file)return;
const reader=new FileReader();
reader.onload=ev=>{{document.getElementById('previewImg').src=ev.target.result;document.getElementById('preview').style.display='block';}};
reader.readAsDataURL(file);
document.getElementById('loading').style.display='block';
const fd=new FormData();fd.append('file',file);fd.append('prompt','Describe the image.');
const r=await fetch('/api/analyze',{method:'POST',body:fd});
const html=await r.text();
const w=window.open('','_blank');w.document.write(html);w.document.close();
document.getElementById('loading').style.display='none';
};
</script></body></html>"""


@app.post("/api/analyze", response_class=HTMLResponse)
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

    return generate_dashboard_html(img_b64, result, prompt)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
