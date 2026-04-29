from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.unet.load_adapter("lora_model")

image = pipe("robot en hospital futurista, ultra realista").images[0]
image.save("resultado.png")