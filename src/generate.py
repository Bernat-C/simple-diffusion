import torch
import matplotlib.pyplot as plt
from model import SimpleUnet
from torchvision.utils import save_image
from pathlib import Path
from config import load_config_yaml

config = load_config_yaml("../conf/config1.yaml")
    
# --- Generation settings ---
IMG_SIZE = config['generation']['image_size']
N_SAMPLES = config['generation']['num_samples_per_class']
DEVICE = config["generation"]["auto"] if config["generation"]["auto"] != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = config['generation']['model_path']
OUTPUT_PATH = config['generation']['generation_output_dir']

# --- Diffusion settings (must match training) ---
T = config['diffusion']['timesteps']
beta_min = config['diffusion']['beta_start']
beta_max = config['diffusion']['beta_end']
betas = torch.linspace(beta_min, beta_max, T)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

@torch.no_grad()
def p_sample(model, x, t, t_index, y):
    # Get model prediction
    # To multiply, we need the tensors to be [BatchSize, Channels, Height, Width]
    betas_t = betas[t_index].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t_index]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
    sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas[t_index]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(DEVICE)

    # Equation 11 from DDPM paper
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, y) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = posterior_variance[t_index].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(DEVICE)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample(model, image_size, batch_size, channels, y):
    # Start with pure noise
    img = torch.randn((batch_size, channels, image_size, image_size), device=DEVICE)

    # Denoise step by step
    for i in reversed(range(0, T)):
        t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
        img = p_sample(model, img, t, i, y)

    # Rescale from [-1, 1] to [0, 1]
    img = (img + 1) * 0.5
    return img

def generate(i):
    # Load the trained model
    model = SimpleUnet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    generated_images = []
    print(f"Generating digit {i}...")
    label = torch.tensor([i] * N_SAMPLES, device=DEVICE) # Generate N_SAMPLES of each digit
    images = sample(model, IMG_SIZE, N_SAMPLES, 1, label)
    generated_images.append(images)

    # Combine images into a grid
    all_images = torch.cat(generated_images)
    save_image(all_images, Path(OUTPUT_PATH) / f"mnist_digits_{i:00}.png", nrow=N_SAMPLES)
    print("Generated digits saved as mnist_digits.png")

if __name__ == "__main__":
    for digit in range(10):
        generate(digit)