const API = {
    async health() {
        const r = await fetch('/api/health');
        return r.json();
    },

    async analyze(file, prompt) {
        const fd = new FormData();
        fd.append('file', file);
        fd.append('prompt', prompt);
        const r = await fetch('/api/analyze', { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    },

    async analyzeJson(file, prompt) {
        const fd = new FormData();
        fd.append('file', file);
        fd.append('prompt', prompt);
        const r = await fetch('/api/analyze/json', { method: 'POST', body: fd });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    },

    async listRuns(limit = 50) {
        const r = await fetch(`/api/runs?limit=${limit}`);
        return r.json();
    },

    async getRun(runId) {
        const r = await fetch(`/api/runs/${runId}`);
        return r.json();
    },

    async deleteRun(runId) {
        const r = await fetch(`/api/runs/${runId}`, { method: 'DELETE' });
        return r.json();
    },

    async listModels() {
        const r = await fetch('/api/models');
        return r.json();
    }
};

function formatTime(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    return d.toLocaleString();
}

function statusBadge(status) {
    const cls = { completed: 'badge-success', failed: 'badge-error', running: 'badge-warning' };
    return `<span class="badge ${cls[status] || 'badge-info'}">${status}</span>`;
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 24px; right: 24px; padding: 12px 20px;
        background: var(--bg-tertiary); border: 1px solid var(--border);
        border-radius: 10px; font-size: 13px; z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function initUploadZone(zone, input, preview, callback) {
    if (!zone) return;

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0], input, preview, callback);
        }
    });

    input.addEventListener('change', e => {
        if (e.target.files.length) {
            handleFile(e.target.files[0], input, preview, callback);
        }
    });
}

function handleFile(file, input, preview, callback) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = e => {
        if (preview) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);

    if (callback) callback(file);
}
