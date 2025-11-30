import torch
from model import Generator
import matplotlib.pyplot as plt
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize generator
netG = Generator()
netG.load_state_dict(torch.load('MNIST_cDCGAN_generator_param.pkl', map_location=device))
netG.to(device)
netG.eval()

# Generate a batch of 10 digits
z = torch.randn(10, 100, 1, 1).to(device)
labels = torch.zeros(10, 10, 1, 1).to(device)
labels.scatter_(1, torch.arange(10).view(-1,1,1,1).to(torch.long), 1)

with torch.no_grad():
    fake_images = netG(z, labels)  # Shape: [10,1,32,32]

# -----------------------------
# Display images in a grid
# -----------------------------
fig, axes = plt.subplots(1, 10, figsize=(12,2))
for i in range(10):
    axes[i].imshow(fake_images[i, 0].cpu().numpy(), cmap='gray')
    axes[i].axis('off')
plt.tight_layout()
plt.show()
