# Anime Image Generator

A small Gradio app for generating anime-style images. Railway serves the UI; image generation runs remotely through Hugging Face Inference Providers and fal.

```text
Browser → Railway / Gradio → Hugging Face Router → fal GPU → image
```

The application does not run SDXL, Diffusers, or LoRA weights locally. It uses:

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- LoRA: `serenitymea/Myanime_model` / `Anime.safetensors`
- Provider route: `fal-ai/fast-sdxl`

## Configuration

Copy `.env.example` to `.env` and set `HF_TOKEN`:

```env
HF_TOKEN=hf_your_token_here
IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
IMAGE_API_URL=https://router.huggingface.co/fal-ai/fal-ai/fast-sdxl
LORA_WEIGHT_NAME=Anime.safetensors
IMAGE_API_TIMEOUT=120
GRADIO_CONCURRENCY=1
LOG_LEVEL=INFO
```

The token needs the **Inference Providers** permission. `PORT` is supplied by Railway and should not be set there.

Each generation sends a prompt, SDXL settings, and (when enabled) this LoRA input to fal:

```json
{
  "loras": [{
    "path": "https://huggingface.co/serenitymea/Myanime_model/resolve/main/Anime.safetensors",
    "scale": 0.8
  }]
}
```

The LoRA strength slider controls `scale`. fal downloads the public LoRA directly; Railway never stores or downloads model weights. The fal safety-checker option is disabled in the request, though provider-level policy may still apply.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Open `http://localhost:7860`.

## Deploy to Railway

1. Push the project to GitHub.
2. Create a Railway project from the repository.
3. Add the variables from `.env.example`, especially `HF_TOKEN`.
4. Deploy. Railway uses the included `railway.json` and Dockerfile.

## Checks

```powershell
pytest -q
```

If generation returns HTTP 402, the Hugging Face account has used its included monthly Inference Providers credits. Add credits, use a plan with more included usage, or wait for the next renewal.
