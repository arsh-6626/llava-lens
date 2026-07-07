import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from llava_lens.core.config import Config, get_config
from llava_lens.core.logging import get_logger
from llava_lens.models.base import BaseModelWrapper
from llava_lens.analysis.pipeline import AnalysisPipeline
from llava_lens.storage.local import LocalStorage
from llava_lens.tracking.tracker import ExperimentTracker
from llava_lens.visualization.dashboard import DashboardGenerator

logger = get_logger(__name__)

app_instance: Optional[FastAPI] = None
model: Optional[BaseModelWrapper] = None
pipeline: Optional[AnalysisPipeline] = None
storage: Optional[LocalStorage] = None
tracker: Optional[ExperimentTracker] = None
dashboard_gen: Optional[DashboardGenerator] = None


def create_app(config: Optional[Config] = None) -> FastAPI:
    global app_instance, model, pipeline, storage, tracker, dashboard_gen

    if config is None:
        config = get_config()

    app_instance = FastAPI(
        title="LLAVA-LENS",
        description="Logit Lens analysis for Vision-Language Models",
        version="2.0.0",
    )

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=config.web.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    storage = LocalStorage(config)
    tracker = ExperimentTracker(config)
    dashboard_gen = DashboardGenerator(config)

    @app_instance.on_event("startup")
    async def startup():
        global model, pipeline
        from llava_lens.core.registry import get_registry
        registry = get_registry()
        model_cls = registry.get("models", "llava")
        model = model_cls(config)
        model.load()
        pipeline = AnalysisPipeline(config, model)
        logger.info("Model loaded and pipeline ready")

    @app_instance.get("/", response_class=HTMLResponse)
    async def index():
        return get_web_ui_html()

    @app_instance.get("/api/health")
    async def health():
        return {"status": "ok", "model_loaded": model is not None and model._loaded}

    @app_instance.get("/api/models")
    async def list_models():
        from llava_lens.core.registry import get_registry
        registry = get_registry()
        return {"models": registry.list_available("models")}

    @app_instance.post("/api/analyze")
    async def analyze_image(
        file: UploadFile = File(...),
        prompt: str = "Describe the image.",
        layer_idx: int = -1,
    ):
        if model is None or pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        run = tracker.create_run(
            name=f"web_{file.filename}",
            model_name=config.model.name,
            config={"prompt": prompt, "filename": file.filename},
        )

        try:
            result = pipeline.analyze(image, prompt, layer_idx)
            run.set_status("completed")
            run.log_metric("num_patches_selected", len(result.patch_selection.selected_indices))
            tracker.save_run(run)

            html = dashboard_gen.generate(
                image=image,
                logit_lens_result=result.logit_lens,
                prompt=prompt,
                model_name=config.model.name,
            )

            return HTMLResponse(content=html)

        except Exception as e:
            run.set_status("failed")
            tracker.save_run(run)
            raise HTTPException(status_code=500, detail=str(e))

    @app_instance.post("/api/analyze/json")
    async def analyze_image_json(
        file: UploadFile = File(...),
        prompt: str = "Describe the image.",
        layer_idx: int = -1,
    ):
        if model is None or pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        try:
            result = pipeline.analyze(image, prompt, layer_idx)

            patch_info = []
            for idx in result.patch_selection.selected_indices:
                if idx < len(result.logit_lens.token_labels):
                    label = result.logit_lens.token_labels[idx]
                    conf = result.patch_selection.confidences.get(idx, 0)
                    patch_info.append({"index": idx, "label": label, "confidence": conf})

            return JSONResponse({
                "status": "success",
                "model": config.model.name,
                "prompt": prompt,
                "num_layers": result.logit_lens.num_layers,
                "selected_patches": patch_info,
                "num_high_confidence": len(result.patch_selection.high_confidence_indices),
                "num_low_confidence": len(result.patch_selection.low_confidence_indices),
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app_instance.get("/api/runs")
    async def list_runs(limit: int = 50):
        runs = tracker.list_runs(limit)
        return {"runs": [run.to_dict() for run in runs]}

    @app_instance.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        run = tracker.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.to_dict()

    @app_instance.get("/api/runs/{run_id}/metrics")
    async def get_run_metrics(run_id: str):
        metrics = tracker.get_run_metrics(run_id)
        return {"metrics": [{"name": m.name, "value": m.value, "step": m.step} for m in metrics]}

    @app_instance.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str):
        success = tracker.delete_run(run_id)
        if not success:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": "deleted"}

    return app_instance


def get_web_ui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLAVA-LENS</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', system-ui, sans-serif; background: #0f172a; color: #f1f5f9; min-height: 100vh; }
        .header { background: #1e293b; border-bottom: 1px solid #334155; padding: 20px 40px; display: flex; align-items: center; justify-content: space-between; }
        .header h1 { font-size: 24px; font-weight: 700; }
        .header .status { font-size: 12px; color: #94a3b8; padding: 6px 12px; background: #0f172a; border-radius: 20px; }
        .container { max-width: 1200px; margin: 40px auto; padding: 0 20px; }
        .upload-card { background: #1e293b; border: 2px dashed #334155; border-radius: 12px; padding: 60px; text-align: center; transition: all 0.3s; cursor: pointer; }
        .upload-card:hover { border-color: #60a5fa; background: #1e293b; }
        .upload-card.dragover { border-color: #60a5fa; background: rgba(96, 165, 250, 0.1); }
        .upload-icon { font-size: 48px; margin-bottom: 16px; }
        .upload-text { font-size: 18px; margin-bottom: 8px; }
        .upload-hint { font-size: 14px; color: #94a3b8; }
        .controls { display: flex; gap: 16px; margin-top: 24px; }
        .input-group { flex: 1; }
        .input-group label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
        .input-group input, .input-group select { width: 100%; padding: 12px; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #f1f5f9; font-size: 14px; }
        .input-group input:focus, .input-group select:focus { outline: none; border-color: #60a5fa; }
        .btn { padding: 12px 24px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .btn-primary { background: #60a5fa; color: #0f172a; }
        .btn-primary:hover { background: #93bbfd; }
        .btn-primary:disabled { background: #475569; cursor: not-allowed; }
        .preview { margin-top: 24px; display: none; }
        .preview img { max-width: 100%; border-radius: 8px; border: 1px solid #334155; }
        .result { margin-top: 24px; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; }
        .stat-value { font-size: 24px; font-weight: 700; color: #60a5fa; }
        .stat-label { font-size: 12px; color: #94a3b8; margin-top: 4px; }
        .runs-list { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; }
        .runs-list h3 { margin-bottom: 12px; font-size: 14px; color: #94a3b8; }
        .run-item { padding: 12px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
        .run-item:last-child { border-bottom: none; }
        .run-id { font-family: monospace; font-size: 12px; color: #60a5fa; }
        .run-status { font-size: 12px; padding: 4px 8px; border-radius: 4px; }
        .run-status.completed { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
        .run-status.failed { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
        .run-status.running { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
        #loading { display: none; text-align: center; padding: 40px; }
        .spinner { border: 3px solid #334155; border-top: 3px solid #60a5fa; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 16px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>LLAVA-LENS</h1>
        <div class="status" id="status">Checking...</div>
    </div>
    <div class="container">
        <div class="upload-card" id="uploadCard">
            <div class="upload-icon">&#128444;</div>
            <div class="upload-text">Drop an image here or click to upload</div>
            <div class="upload-hint">Supports JPG, PNG, BMP</div>
            <input type="file" id="fileInput" accept="image/*" style="display:none">
        </div>
        <div class="controls">
            <div class="input-group">
                <label>Prompt</label>
                <input type="text" id="prompt" value="Describe the image." placeholder="Enter your prompt...">
            </div>
            <div class="input-group">
                <label>Layer Index</label>
                <input type="number" id="layerIdx" value="-1" min="-32" max="32">
            </div>
            <div style="display:flex; align-items:flex-end;">
                <button class="btn btn-primary" id="analyzeBtn" disabled>Analyze</button>
            </div>
        </div>
        <div class="preview" id="preview">
            <img id="previewImg" src="" alt="Preview">
        </div>
        <div id="loading">
            <div class="spinner"></div>
            <div>Analyzing image...</div>
        </div>
        <div class="result" id="result"></div>
    </div>
    <script>
        const uploadCard = document.getElementById('uploadCard');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const preview = document.getElementById('preview');
        const previewImg = document.getElementById('previewImg');
        const loading = document.getElementById('loading');
        const resultDiv = document.getElementById('result');
        const statusDiv = document.getElementById('status');
        let selectedFile = null;

        fetch('/api/health').then(r => r.json()).then(d => {
            statusDiv.textContent = d.model_loaded ? 'Model Ready' : 'Loading Model...';
            statusDiv.style.color = d.model_loaded ? '#22c55e' : '#f59e0b';
        }).catch(() => { statusDiv.textContent = 'Offline'; statusDiv.style.color = '#ef4444'; });

        uploadCard.addEventListener('click', () => fileInput.click());
        uploadCard.addEventListener('dragover', (e) => { e.preventDefault(); uploadCard.classList.add('dragover'); });
        uploadCard.addEventListener('dragleave', () => uploadCard.classList.remove('dragover'));
        uploadCard.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadCard.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => { previewImg.src = e.target.result; preview.style.display = 'block'; };
            reader.readAsDataURL(file);
            analyzeBtn.disabled = false;
        }

        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            loading.style.display = 'block';
            resultDiv.innerHTML = '';
            preview.style.display = 'none';

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('prompt', document.getElementById('prompt').value);
            formData.append('layer_idx', document.getElementById('layerIdx').value);

            try {
                const response = await fetch('/api/analyze', { method: 'POST', body: formData });
                if (!response.ok) throw new Error(await response.text());
                const html = await response.text();
                const iframe = document.createElement('iframe');
                iframe.style.width = '100%';
                iframe.style.height = '700px';
                iframe.style.border = '1px solid #334155';
                iframe.style.borderRadius = '8px';
                resultDiv.appendChild(iframe);
                iframe.contentDocument.open();
                iframe.contentDocument.write(html);
                iframe.contentDocument.close();
            } catch (err) {
                resultDiv.innerHTML = '<div style="color:#ef4444;padding:20px;">Error: ' + err.message + '</div>';
            } finally {
                loading.style.display = 'none';
            }
        });

        fetch('/api/runs?limit=10').then(r => r.json()).then(d => {
            if (d.runs && d.runs.length) {
                let html = '<div class="runs-list"><h3>Recent Runs</h3>';
                d.runs.forEach(run => {
                    html += '<div class="run-item"><span class="run-id">' + run.run_id + '</span><span>' + (run.name || 'unnamed') + '</span><span class="run-status ' + run.status + '">' + run.status + '</span></div>';
                });
                html += '</div>';
                resultDiv.innerHTML = html;
            }
        });
    </script>
</body>
</html>"""
