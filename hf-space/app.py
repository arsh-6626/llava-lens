import base64
from io import BytesIO

import spaces
import torch
import gradio as gr
from PIL import Image

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
IMAGE_TOKEN_ID = 32000
IMAGE_SIZE = 336
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

model = None
processor = None


def load_model():
    global model, processor
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)


@spaces.GPU(duration=120)
def analyze_image_gpu(image, prompt):
    global model, processor
    if model is None:
        load_model()

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

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

    return {"token_labels": token_labels, "image_mask": image_mask, "layer_data": all_layer_data, "num_layers": num_layers}


def build_dashboard(image_b64, result, prompt):
    tl = result["token_labels"]
    im = result["image_mask"]
    ld = result["layer_data"]
    nl = result["num_layers"]
    gs = IMAGE_SIZE // PATCH_SIZE

    tr = ""
    ir = ""
    for t, label in enumerate(tl):
        cells = ""
        for l in range(nl):
            top = ld[l][t][0]
            p = top["p"]
            a = min(0.6, p * 0.6)
            c = "#fff" if p > 0.8 else "#94a3b8" if p > 0.3 else "#475569"
            cells += '<td style="background:rgba(99,102,241,%.2f);color:%s;font-size:11px;padding:4px 6px;" title="L%d: %s (%.1f%%)">%s</td>' % (a, c, l+1, top["t"], p*100, top["t"])
        if im[t]:
            ir += '<tr class="pr" data-l="%s"><td style="padding:4px 8px;"><span style="background:rgba(239,68,68,0.15);color:#f87171;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;">%s</span></td>%s</tr>' % (label, label, cells)
        else:
            tr += '<tr><td style="padding:4px 8px;font-weight:500;">%s</td>%s</tr>' % (label, cells)

    lh = "".join(['<th style="padding:8px;font-size:11px;">L%d</th>' % (i+1) for i in range(nl)])

    return '<!DOCTYPE html><html><head><meta charset="UTF-8"><style>@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");*{margin:0;padding:0;box-sizing:border-box;}body{font-family:"Inter",system-ui,sans-serif;background:#09090b;color:#fafafa;overflow:hidden;height:100vh;animation:fadeIn .4s ease;}@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}.wrap{display:flex;height:100vh;}.side{width:380px;background:#111113;border-right:1px solid rgba(255,255,255,0.06);display:flex;flex-direction:column;overflow-y:auto;animation:slideLeft .5s ease;}@keyframes slideLeft{from{transform:translateX(-40px);opacity:0}to{transform:translateX(0);opacity:1}}.logo{padding:24px;border-bottom:1px solid rgba(255,255,255,0.06);}.logo h1{font-size:20px;font-weight:700;background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}.logo .sub{font-size:12px;color:#52525b;margin-top:4px;font-family:monospace;}.img-area{padding:20px;flex:1;display:flex;flex-direction:column;align-items:center;overflow-y:auto;}.img-frame{position:relative;width:336px;height:336px;border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);box-shadow:0 25px 50px rgba(0,0,0,0.5);animation:scaleIn .4s ease;}@keyframes scaleIn{from{transform:scale(0.9);opacity:0}to{transform:scale(1);opacity:1}}.img-frame img{width:100%;height:100%;object-fit:cover;}.patch-hl{position:absolute;border:2px solid #ef4444;background:rgba(239,68,68,0.2);pointer-events:none;display:none;box-shadow:0 0 20px rgba(239,68,68,0.5);border-radius:2px;z-index:10;}.info{margin-top:16px;padding:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;width:100%;font-size:13px;color:#a1a1aa;}.info strong{color:#e2e8f0;}.main{flex:1;display:flex;flex-direction:column;overflow:hidden;animation:fadeIn .6s ease;}.tbl-hdr{padding:16px 24px;border-bottom:1px solid rgba(255,255,255,0.06);display:flex;align-items:center;justify-content:space-between;background:#111113;}.tbl-hdr h2{font-size:14px;font-weight:600;}.tabs{display:flex;gap:4px;}.tab{padding:6px 14px;border-radius:6px;font-size:12px;font-weight:500;cursor:pointer;background:transparent;color:#71717a;border:1px solid transparent;transition:all .2s;}.tab:hover{color:#fafafa;background:rgba(255,255,255,0.05);}.tab.active{background:rgba(99,102,241,0.15);color:#818cf8;border-color:rgba(99,102,241,0.3);}.tbl-wrap{flex:1;overflow:auto;}table{border-collapse:collapse;width:100%;font-size:12px;}th,td{padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.03);border-right:1px solid rgba(255,255,255,0.03);white-space:nowrap;text-align:left;}thead th{position:sticky;top:0;background:#09090b;z-index:10;font-weight:600;color:#52525b;font-size:11px;padding:8px 10px;border-bottom:2px solid rgba(255,255,255,0.08);}.rh{position:sticky;left:0;background:#09090b;z-index:5;text-align:right;min-width:110px;border-right:2px solid rgba(255,255,255,0.08);color:#d4d4d8;font-weight:500;}tr:hover td{background:rgba(99,102,241,0.05);}td:hover{outline:1px solid #818cf8;outline-offset:-1px;z-index:1;position:relative;}.ttip{position:fixed;background:#18181b;color:white;padding:14px;border-radius:10px;font-size:12px;pointer-events:none;z-index:1000;display:none;border:1px solid rgba(255,255,255,0.1);box-shadow:0 25px 50px rgba(0,0,0,0.8);min-width:220px;}.ttip-h{font-weight:600;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.08);color:#818cf8;}.ttip-r{display:flex;justify-content:space-between;margin-bottom:4px;}.ttip-r span:last-child{color:#52525b;}.ttip-b{height:3px;background:rgba(255,255,255,0.06);border-radius:2px;margin-bottom:8px;}.ttip-f{height:100%;background:linear-gradient(90deg,#6366f1,#818cf8);border-radius:2px;}</style></head><body><div class="wrap"><div class="side"><div class="logo"><h1>LLAVA-LENS</h1><div class="sub">llava-1.5-7b-hf</div></div><div class="img-area"><div class="img-frame"><img src="data:image/png;base64,%s"><div class="patch-hl" id="phl"></div></div><div class="info"><strong>Prompt:</strong> %s</div><div style="margin-top:12px;padding:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;width:100%%;font-size:12px;color:#71717a;"><div>Hover cells for predictions</div><div>Blue = confidence level</div><div>IMG = image patches</div></div></div></div><div class="main"><div class="tbl-hdr"><h2>Token Predictions Across Layers</h2><div class="tabs"><div class="tab active" onclick="showTab(\'all\')">All</div><div class="tab" onclick="showTab(\'text\')">Text</div><div class="tab" onclick="showTab(\'image\')">Image</div></div></div><div class="tbl-wrap"><table><thead><tr><th class="rh">Token</th>%s</tr></thead><tbody id="tb">%s<tr id="sep" style="display:none;"><td colspan="%d" style="padding:4px;background:rgba(239,68,68,0.05);border-bottom:1px solid rgba(239,68,68,0.2);"><span style="font-size:10px;color:#f87171;text-transform:uppercase;letter-spacing:1px;">Image Patches</span></td></tr>%s</tbody></table></div></div></div><div class="ttip" id="ttip"></div><script>const GS=%d,PS=%d;document.querySelectorAll("td").forEach(td=>{td.onmouseenter=function(){const t=this.getAttribute("title");if(!t)return;const p=t.split(": ");const l=p[0];const r=p[1]?p[1].split(" ("):["",""];const tk=r[0];const pr=r[1]?r[1].replace(")",""):"";let h="<div class=\\"ttip-h\\">"+l+"</div><div class=\\"ttip-r\\"><span>"+tk+"</span><span>"+pr+"</span></div>";const pct=parseFloat(pr)||0;h+="<div class=\\"ttip-b\\"><div class=\\"ttip-f\\" style=\\"width:"+pct+"%\\"></div></div>";const tt=document.getElementById("ttip");tt.innerHTML=h;tt.style.display="block";const rc=this.getBoundingClientRect();let left=rc.right+12,top=rc.top;if(left+240>innerWidth)left=rc.left-252;if(top+150>innerHeight)top=innerHeight-150;tt.style.left=left+"px";tt.style.top=top+"px";this.style.outline="1px solid #818cf8";};td.onmouseleave=function(){document.getElementById("ttip").style.display="none";this.style.outline="";};});document.querySelectorAll(".pr").forEach(tr=>{tr.onmouseenter=function(){const l=this.dataset.l;const idx=parseInt(l.replace("<IMG","").replace(">",""))-1;const r=Math.floor(idx/GS),c=idx%%GS;const hl=document.getElementById("phl");hl.style.width=PS+"px";hl.style.height=PS+"px";hl.style.left=(c*PS)+"px";hl.style.top=(r*PS)+"px";hl.style.display="block";};tr.onmouseleave=()=>document.getElementById("phl").style.display="none";});function showTab(t){document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));event.target.classList.add("active");const sep=document.getElementById("sep");const pr=document.querySelectorAll(".pr");const tr=document.querySelectorAll("#tb tr:not(.pr):not(#sep)");if(t==="all"){sep.style.display="none";tr.forEach(r=>r.style.display="");pr.forEach(r=>r.style.display="");}else if(t==="text"){sep.style.display="none";tr.forEach(r=>r.style.display="");pr.forEach(r=>r.style.display="none");}else if(t==="image"){sep.style.display="";tr.forEach(r=>r.style.display="none");pr.forEach(r=>r.style.display="");}}</script></body></html>' % (image_b64, prompt, lh, tr, nl+1, ir, gs, PATCH_SIZE)


with gr.Blocks(
    css="@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');*{font-family:'Inter',system-ui,sans-serif!important;}.gradio-container{max-width:100%!important;padding:0!important;background:#09090b!important;min-height:100vh;}.gr-blocks{padding:0!important;background:#09090b!important;border:none!important;}footer{display:none!important;}.gr-button-primary{background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;border:none!important;font-weight:600!important;padding:14px 28px!important;}.gr-button-primary:hover{background:linear-gradient(135deg,#818cf8,#a78bfa)!important;transform:translateY(-1px);}.gr-input,.gr-textbox{background:#18181b!important;border:1px solid rgba(255,255,255,0.08)!important;color:#fafafa!important;border-radius:10px!important;}.gr-input:focus,.gr-textbox:focus{border-color:#6366f1!important;box-shadow:0 0 0 2px rgba(99,102,241,0.2)!important;}.gr-label{color:#a1a1aa!important;font-size:13px!important;font-weight:500!important;}.gr-image{background:#18181b!important;border-radius:12px!important;border:1px solid rgba(255,255,255,0.08)!important;}.gr-html{background:transparent!important;border:none!important;padding:0!important;}.gr-row{gap:0!important;padding:0!important;}.gr-col{padding:0!important;}.gr-panel{background:transparent!important;border:none!important;}.gr-box{background:transparent!important;border:none!important;}.gr-form{gap:0!important;}.gr-markdown{color:#fafafa!important;}.gr-markdown h1{color:#fafafa!important;font-size:32px!important;font-weight:700!important;letter-spacing:-1px!important;}.gr-markdown p{color:#a1a1aa!important;}.gr-markdown a{color:#818cf8!important;}.gr-prose{color:#fafafa!important;}#component-2{padding:32px!important;}""",
    title="LLAVA-LENS",
) as demo:
    gr.Markdown("""
# 🔬 LLAVA-LENS
**Logit Lens analysis for Vision-Language Models** — See how tokens emerge across transformer layers.
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=420):
            gr.Markdown("### Upload & Configure")
            image_input = gr.Image(type="pil", label="Image", height=350, sources=["upload", "clipboard"])
            with gr.Row():
                prompt_input = gr.Textbox(
                    value="Describe the image.",
                    label="Prompt",
                    placeholder="Enter your prompt...",
                    scale=3,
                )
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)

        with gr.Column(scale=2):
            result_html = gr.HTML(
                value="""
                <div style='display:flex;align-items:center;justify-content:center;height:calc(100vh - 200px);color:#52525b;flex-direction:column;gap:16px;'>
                    <div style='width:80px;height:80px;border-radius:20px;background:rgba(99,102,241,0.1);display:flex;align-items:center;justify-content:center;'>
                        <svg width='40' height='40' viewBox='0 0 24 24' fill='none' stroke='#818cf8' stroke-width='1.5' style='opacity:0.6;'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>
                    </div>
                    <div style='font-size:18px;font-weight:600;'>Ready to Analyze</div>
                    <div style='font-size:14px;'>Upload an image and click Analyze</div>
                </div>
                """,
            )

    def run_analysis(image, prompt):
        if image is None:
            return "<div style='text-align:center;padding:80px;color:#ef4444;'>Please upload an image</div>"
        result = analyze_image_gpu(image, prompt)
        buffered = BytesIO()
        image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        return build_dashboard(img_b64, result, prompt)

    analyze_btn.click(fn=run_analysis, inputs=[image_input, prompt_input], outputs=[result_html])
