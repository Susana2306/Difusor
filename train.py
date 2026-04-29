import os
import random
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
output_dir = "lora_model"
epochs = 3
lr = 1e-4
batch_size = 2
gradient_accumulation_steps = 4
max_samples = 5000  # None para usar todo el dataset (~40k filas: 8k imágenes × 5 captions)
image_size = 512

print(f"Usando dispositivo: {device}")

class Flickr8kDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform, caption_cols):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.caption_cols = caption_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"].convert("RGB")
        image = self.transform(image)
        col = random.choice(self.caption_cols)
        caption = sample[col]
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0),
        }

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# =========================
# CARGAR MODELO BASE
# =========================
print("Cargando modelo base...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
)
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Congelar VAE y text encoder
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

# =========================
# CONFIG LoRA
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.1,
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.print_trainable_parameters()

pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)

# =========================
# CARGAR DATASET
# =========================
print("Cargando dataset Flickr8k...")
raw_dataset = load_dataset("jxie/flickr8k", split="train")

if max_samples:
    raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))

print(f"Muestras a usar: {len(raw_dataset)}")

# Detectar columnas de caption automáticamente
sample_keys = list(raw_dataset[0].keys())
print(f"Columnas del dataset: {sample_keys}")
caption_cols = [c for c in sample_keys if c.startswith("caption")]
if not caption_cols:
    caption_cols = [c for c in sample_keys if c in ("text", "sentence", "sentences", "captions")]
if not caption_cols:
    raise ValueError(f"No se encontró columna de caption. Columnas disponibles: {sample_keys}")
print(f"Columnas de caption detectadas: {caption_cols}")

dataset = Flickr8kDataset(raw_dataset, pipe.tokenizer, transform, caption_cols)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device == "cuda"),
)

# =========================
# OPTIMIZADOR Y SCALER
# =========================
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr, weight_decay=1e-2)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda") if use_amp else None

# =========================
# ENTRENAMIENTO
# =========================
pipe.unet.train()

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            encoder_hidden_states = pipe.text_encoder(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if use_amp:
            with torch.amp.autocast("cuda"):
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            scaler.scale(loss / gradient_accumulation_steps).backward()
        else:
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            (loss / gradient_accumulation_steps).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Loss promedio epoch {epoch+1}: {avg_loss:.4f}")

# =========================
# GUARDAR MODELO
# =========================
os.makedirs(output_dir, exist_ok=True)
pipe.unet.save_pretrained(output_dir)
print(f"\nModelo guardado en: {output_dir}")
