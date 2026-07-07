# LLAVA-LENS

Logit Lens analysis for Vision-Language Models - Research Platform

## Overview

LLAVA-LENS is a research platform for analyzing intermediate transformer representations in Vision-Language Models (VLMs) using the Logit Lens technique. Instead of relying only on final predictions, it inspects how semantic concepts emerge across layers, improving classification tasks where subtle visual cues matter (e.g., trauma detection).

## Features

- **Multi-Model Support**: Plug any VLM via model registry
- **Logit Lens Analysis**: Track token predictions across transformer layers
- **Attention Analysis**: Visualize and aggregate attention maps
- **Patch Selection**: Identify high-confidence image patches
- **Batch Processing**: Parallel processing of large image datasets
- **Web Interface**: Interactive dashboard with real-time analysis
- **Experiment Tracking**: SQLite-based tracking of runs and metrics
- **Storage Management**: Organized local filesystem with caching

## Quick Start

### Installation

```bash
pip install -e .
```

### Web Interface

```bash
llava-lens serve
```

Open http://localhost:8000 in your browser.

### CLI Usage

```bash
llava-lens analyze image.jpg -p "Describe the image."
llava-lens batch ./images -p "Describe the image." -o ./results
llava-lens runs
llava-lens models
```

### Python API

```python
from llava_lens.core.config import Config
from llava_lens.core.registry import get_registry
from llava_lens.analysis.pipeline import AnalysisPipeline

config = Config()
registry = get_registry()
model_cls = registry.get("models", "llava")
model = model_cls(config)
model.load()

pipeline = AnalysisPipeline(config, model)
result = pipeline.analyze("image.jpg", "Describe the image.")
```

### Docker

```bash
docker-compose up
```

## Project Structure

```
llava-lens/
├── src/llava_lens/
│   ├── core/           # Registry, config, logging, exceptions
│   ├── models/         # VLM wrappers (LLaVA, custom)
│   ├── analysis/       # Logit lens, attention, patch selection
│   ├── data/           # Dataset handling, transforms
│   ├── processing/     # Batch processing pipeline
│   ├── storage/        # Local filesystem, caching
│   ├── tracking/       # Experiment tracking (SQLite)
│   ├── visualization/  # HTML dashboards
│   ├── web/            # FastAPI web interface
│   └── cli/            # Click CLI commands
├── configs/            # YAML configuration files
├── examples/           # Usage examples
├── data/               # Data storage
├── Dockerfile
└── docker-compose.yml
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  name: "llava-hf/llava-1.5-7b-hf"
  device: "auto"
  image_size: 336

analysis:
  logit_lens:
    top_k: 5
    confidence_threshold: 0.01

web:
  host: "0.0.0.0"
  port: 8000
```

## Adding Custom Models

```python
from llava_lens.core.registry import get_registry
from llava_lens.models.base import BaseModelWrapper

@get_registry().register_model("my-model")
class MyModelWrapper(BaseModelWrapper):
    def load(self):
        # Load your model
        pass

    def forward(self, image, prompt, **kwargs):
        # Run inference
        pass
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `llava-lens analyze` | Analyze a single image |
| `llava-lens batch` | Process a directory of images |
| `llava-lens serve` | Start web interface |
| `llava-lens runs` | List experiment runs |
| `llava-lens models` | List available models |
| `llava-lens config` | View/set configuration |
| `llava-lens cache-clear` | Clear cache |
| `llava-lens storage-cleanup` | Clean storage directories |

## License

MIT
