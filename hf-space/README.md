---
title: LLAVA-LENS
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
license: mit
resources:
  gpu: T4
---

# LLAVA-LENS

**Research Platform for Vision-Language Model Analysis**

Logit Lens analysis, attention visualization, and experiment tracking for VLMs.

## Features

- Logit Lens analysis across transformer layers
- Attention map visualization
- Patch selection with confidence scoring
- Batch processing for large datasets
- Experiment tracking with SQLite
- Model registry for pluggable VLMs
- Interactive web dashboard

## Usage

Visit the live demo or run locally:

```bash
pip install -e .
llava-lens serve
```
