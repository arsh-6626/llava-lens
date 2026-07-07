FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir \
    torch transformers accelerate bitsandbytes Pillow numpy \
    pydantic pyyaml click rich fastapi uvicorn jinja2 \
    python-multipart aiofiles tqdm joblib

COPY . .
RUN pip install -e .

EXPOSE 8000

CMD ["llava-lens", "serve", "--host", "0.0.0.0", "--port", "8000"]
