# Anime Image Generator

A production-minded Gradio MVP for generating anime-style images. The web app runs on a small Railway CPU service and sends inference requests to a remote Hugging Face GPU provider. It supports prompt helpers, reproducible seeds, 1–4 images, presets, backend moderation, and clear provider errors.

## Architecture

```text
Browser → Gradio on Railway → safety validation → ImageGenerator interface
                                              → Hugging Face routed inference → PIL image → Gallery
```

Running a diffusion model inside a normal Railway container is impractical: model weights, RAM, and GPU requirements make builds slow and costly. Remote inference keeps the image small and makes `generator/provider.py` replaceable with a Replicate, RunPod, fal.ai, or ComfyUI implementation of `ImageGenerator`.

## Files

```text
.
├── app.py
├── generator/
│   ├── __init__.py
│   ├── base.py
│   └── provider.py
├── safety/
│   ├── __init__.py
├── utils/
│   ├── __init__.py
│   └── prompts.py
├── tests/
│   └── test_prompts.py
├── .env.example
├── .gitignore
├── Dockerfile
├── railway.json
├── pytest.ini
├── requirements.txt
└── README.md
```

## Local setup

Python 3.11 is recommended. Get a Hugging Face user access token with **Inference Providers** permission from `https://huggingface.co/settings/tokens`. Ensure the selected text-to-image model is available through the routed `hf-inference` provider; provider/model availability and pricing can change.

macOS/Linux:

```bash
git clone <your-repository-url>
cd <repository-directory>
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Windows PowerShell:

```powershell
git clone <your-repository-url>
Set-Location <repository-directory>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Edit `.env` and set `IMAGE_API_KEY`. Open `http://localhost:7860`. Never commit `.env`.

### Configuration

| Variable | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face token with Inference Providers permission | empty |
| `IMAGE_MODEL` | Hub model ID | `stabilityai/stable-diffusion-xl-base-1.0` |
| `IMAGE_API_URL` | fal provider route | `https://router.huggingface.co/fal-ai/fal-ai/fast-sdxl` |
| `LORA_WEIGHT_NAME` | Private LoRA filename if the repository has several | empty |
| `IMAGE_API_TIMEOUT` | Request timeout in seconds | `120` |
| `GRADIO_CONCURRENCY` | Simultaneous generation jobs | `1` |
| `PORT` | Listening port | `7860` |

### SDXL with Myanime LoRA through Inference Providers

The app uses the `fal-ai` serverless provider through the Hugging Face Router. The current live SDXL mapping is `fal-ai/fast-sdxl`. `serenitymea/Myanime_model` is public and has a live LoRA mapping to the same provider, with adapter file `Anime.safetensors`. Railway sends only the public Hub URL for that adapter in fal's native SDXL payload:

```json
{
  "prompt": "...",
  "image_size": {"width": 832, "height": 1216},
  "loras": [{"path": "https://huggingface.co/serenitymea/Myanime_model/resolve/main/Anime.safetensors", "scale": 0.8}]
}
```

The actual HTTP endpoint is `https://router.huggingface.co/fal-ai/fal-ai/fast-sdxl`. This is fal's provider ID from Hugging Face's live SDXL mapping; the Router does not accept `/fal-ai/models/<Hub model ID>`. Railway never downloads SDXL or the LoRA: fal downloads the public LoRA URL and runs SDXL on its serverless GPU. Use the **Myanime LoRA** controls to enable the adapter and set its strength.

Run checks with `python -m pytest -q` after installing `pytest`, or use `python -m compileall app.py generator safety utils` for a dependency-free syntax check.

## Deploy to Railway

1. Push this directory to a private or public GitHub repository.
2. In Railway choose **New Project → Deploy from GitHub repo** and select it.
3. Under **Variables**, add `HF_TOKEN` with the **Inference Providers** scope and, if needed, `LORA_WEIGHT_NAME`. Do not set `PORT`; Railway injects it.
4. Railway detects `railway.json`, builds `Dockerfile`, runs `python app.py`, and checks `/`.
5. In **Settings → Networking**, generate a public domain.
6. Check deployment logs for the Gradio listening address, then make one generation request.

No volume, database, GPU, or custom start command is required. A Hobby-sized CPU service is sufficient for the UI because inference is remote. Set a longer health-check/deploy timeout if your region boots slowly.

## Changing the inference provider

Create a class implementing `generator.base.ImageGenerator.generate`, return `GeneratedImage`, then construct it in `get_generator()` in `app.py`. Keep provider exceptions converted to `ProviderError`. The rest of validation, seed display, prompt enhancement, and UI does not change. A custom Hugging Face-compatible endpoint can often be selected using only `IMAGE_API_URL`.

## Safety and MVP limitations

This app is adults-only. It blocks minor/under-18/young-looking terms and sexual requests involving real people on the backend. Fictional, explicitly adult NSFW prompts are not blanket-blocked. The checkbox is an acknowledgement, not age verification.

The local regex filter is deliberately conservative but cannot understand every euphemism, language, celebrity name, or adversarial prompt. Before public deployment, add a dedicated moderation service before inference, provider-side output moderation, authentication, per-user rate limits, abuse reporting, and legally appropriate age assurance. Review the selected provider's acceptable-use policy: it may independently reject NSFW content. Generated images are held in memory and handed to Gradio; the application does not intentionally persist them.

Other MVP constraints: sequential multi-image calls, no job persistence, no accounts, no billing controls, and provider-dependent support for dimensions/steps/seed. If a model rejects a parameter, choose a compatible model or adapt its provider implementation.
