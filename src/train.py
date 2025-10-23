import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

from model import SimpleUnet
from config import load_config_yaml

config = load_config_yaml("../conf/config1.yaml")

# --- Hyperparameters ---
IMG_SIZE = config["general"]["image_size"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["num_epochs"]
DEVICE = config["generation"]["auto"] if config["generation"]["auto"] != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = config["training"]["learning_rate"]
MODEL_PATH = config['generation']['model_path']

# --- Diffusion settings ---
T = config['diffusion']['timesteps']
beta_min = config['diffusion']['beta_start']
beta_max = config['diffusion']['beta_end']
betas = torch.linspace(beta_min, beta_max, T)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# Helper function to get the alpha values for a specific timestep
def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# The forward process (adding noise)
def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(torch.sqrt(alphas_cumprod), t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(torch.sqrt(1. - alphas_cumprod), t, x_0.shape)
    
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def train():
    device = DEVICE
    print(f"Using device: {device}")

    # --- Data Loading ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale to [-1, 1]
    ])
    dataset = MNIST(root="../data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # --- Model and Optimizer ---
    model = SimpleUnet().to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader)
        for step, (images, labels) in enumerate(pbar):
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            x_noisy, noise = forward_diffusion_sample(images, t, device)
            
            # We provide the model with the label information
            predicted_noise = model(x_noisy, t, labels.to(device))
            
            loss = F.mse_loss(noise, predicted_noise)

            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

    # --- Save Model ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

if __name__ == '__main__':
    train()