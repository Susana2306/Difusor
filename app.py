import os
import base64
import time
from io import BytesIO

import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    PixArtAlphaPipeline,
)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel
from transformers import T5Tokenizer

# Suppress the Windows symlinks warning (cosmetic, no impact on functionality)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

app = FastAPI(title="Comparador de Modelos")
templates = Jinja2Templates(directory="templates")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"Device: {device}")

# modelo 1: Stable difussion v1.5 + LoRA
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_LORA_DIR = "lora_model"

print("[SD] Cargando pipeline...")
pipe_sd = StableDiffusionPipeline.from_pretrained(
    SD_MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
)

if os.path.exists(SD_LORA_DIR):
    pipe_sd.unet = PeftModel.from_pretrained(pipe_sd.unet, SD_LORA_DIR)
    pipe_sd.unet.eval()
    print("[SD] LoRA cargado.")
else:
    print("[SD] Sin LoRA — modelo base.")

pipe_sd.scheduler = DPMSolverMultistepScheduler.from_config(pipe_sd.scheduler.config)
pipe_sd.enable_attention_slicing()
if device == "cuda":
    pipe_sd.enable_sequential_cpu_offload()
else:
    pipe_sd.to(device)
print("[SD] Pipeline listo.")


# Segundo modelo: PixArt-alpha DiT + LoRA
PIXART_MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
PIXART_LORA_DIR = "pixart_lora/lora_transformer"
pipe_pixart = None   # se inicializa al primer uso
_pixart_loading = False


def _load_pixart():
    global pipe_pixart, _pixart_loading
    if pipe_pixart is not None:
        return
    _pixart_loading = True
    print("[PixArt] Cargando pipeline (primera vez, puede tardar)...")
    # use_fast=False evita la conversión slow→fast que falla sin sentencepiece
    tokenizer = T5Tokenizer.from_pretrained(PIXART_MODEL_ID, subfolder="tokenizer", use_fast=False)
    pipe = PixArtAlphaPipeline.from_pretrained(
        PIXART_MODEL_ID,
        tokenizer=tokenizer,
        torch_dtype=dtype,
    )
    if os.path.exists(PIXART_LORA_DIR):
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, PIXART_LORA_DIR)
        pipe.transformer.eval()
        print("[PixArt] LoRA cargado.")
    else:
        print("[PixArt] Sin LoRA — modelo base.")
    if device == "cuda":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)
    pipe_pixart = pipe
    _pixart_loading = False
    print("[PixArt] Pipeline listo.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _encode(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Rutas ─────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    sd_steps: int = 15
    sd_guidance: float = 7.5
    pixart_steps: int = 20
    pixart_guidance: float = 4.5


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/generate")
async def generate(body: GenerateRequest):
    if not body.prompt.strip():
        return JSONResponse({"error": "El prompt no puede estar vacío."}, status_code=400)
    steps    = max(5, min(body.sd_steps, 50))
    guidance = max(1.0, min(body.sd_guidance, 20.0))
    with torch.no_grad():
        img = pipe_sd(body.prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
    return JSONResponse({"image": _encode(img)})


@app.post("/compare")
async def compare(body: GenerateRequest):
    if not body.prompt.strip():
        return JSONResponse({"error": "El prompt no puede estar vacío."}, status_code=400)

    _load_pixart()  # no-op si ya está cargado

    sd_steps        = max(5,   min(body.sd_steps,        50))
    sd_guidance     = max(1.0, min(body.sd_guidance,     20.0))
    pixart_steps    = max(5,   min(body.pixart_steps,    30))
    pixart_guidance = max(1.0, min(body.pixart_guidance, 10.0))

    t0 = time.time()
    with torch.no_grad():
        img_sd = pipe_sd(
            body.prompt,
            num_inference_steps=sd_steps,
            guidance_scale=sd_guidance,
        ).images[0]
    sd_time = round(time.time() - t0, 1)

    t1 = time.time()
    with torch.no_grad():
        img_pixart = pipe_pixart(
            body.prompt,
            num_inference_steps=pixart_steps,
            guidance_scale=pixart_guidance,
        ).images[0]
    pixart_time = round(time.time() - t1, 1)

    return JSONResponse({
        "image_sd":     _encode(img_sd),
        "image_pixart": _encode(img_pixart),
        "sd_time":      sd_time,
        "pixart_time":  pixart_time,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
