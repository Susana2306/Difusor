import os
import base64
from io import BytesIO

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from peft import PeftModel
from pydantic import BaseModel

app = FastAPI(title="Generador de Imágenes")
templates = Jinja2Templates(directory="templates")


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
lora_dir = "lora_model"

print(f"Cargando pipeline en {device}...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

if os.path.exists(lora_dir):
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    pipe.unet.eval()
    print("LoRA fine-tuning cargado.")
else:
    print("Sin LoRA — usando modelo base.")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
print("Pipeline listo.")


class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 15
    guidance_scale: float = 7.5


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/generate")
async def generate(body: GenerateRequest):
    if not body.prompt.strip():
        return JSONResponse({"error": "El prompt no puede estar vacío."}, status_code=400)

    steps = max(5, min(body.steps, 50))
    guidance = max(1.0, min(body.guidance_scale, 20.0))

    with torch.no_grad():
        result = pipe(
            body.prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
    image = result.images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    return JSONResponse({"image": img_b64})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
