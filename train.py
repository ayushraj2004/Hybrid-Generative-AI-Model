import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from models import VAE, Generator, Discriminator


# Set up training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 10
input_dim = 784  # For MNIST (28x28 flattened)
hidden_dim = 50
output_dim = input_dim

# Initialize models
vae = VAE(input_dim, latent_dim).to(device)
generator = Generator(latent_dim, output_dim).to(device)
discriminator = Discriminator(input_dim).to(device)

# Optimizers
vae_optimizer = optim.Adam(vae.parameters(), lr=0.001)
gan_optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
gan_optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

# Data Loader (for MNIST)
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

# Define loss functions
def vae_loss(reconstructed, original, mean, log_var):
    reconstruction_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

# Training loop
epochs = 10
for epoch in range(epochs):
    for real_data, _ in train_loader:
        real_data = real_data.view(-1, input_dim).to(device)
        
        # Train VAE
        vae_optimizer.zero_grad()
        reconstructed, mean, log_var = vae(real_data)
        vae_loss_value = vae_loss(reconstructed, real_data, mean, log_var)
        vae_loss_value.backward()
        vae_optimizer.step()
        
        # Generate fake data (detach z)
        z = vae.reparameterize(mean, log_var).detach()
        fake_data = generator(z)

        # Train Discriminator
        gan_optimizer_d.zero_grad()
        real_labels = torch.ones(real_data.size(0), 1).to(device)
        fake_labels = torch.zeros(fake_data.size(0), 1).to(device)
        real_loss = F.binary_cross_entropy(discriminator(real_data), real_labels)
        fake_loss = F.binary_cross_entropy(discriminator(fake_data.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        gan_optimizer_d.step()

        # Train Generator
        gan_optimizer_g.zero_grad()
        g_loss = F.binary_cross_entropy(discriminator(fake_data), real_labels)
        g_loss.backward()
        gan_optimizer_g.step()

    # Print progress
    print(f'Epoch [{epoch+1}/{epochs}], VAE Loss: {vae_loss_value.item():.4f}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}')
    # Function to show generated images
def show_generated_images(images, num_images=10):
    # Reshape images to 28x28 if you're using MNIST dataset
    images = images.view(images.size(0), 28, 28)  # Adjust this if using a different dataset
    grid_size = int(np.ceil(np.sqrt(num_images)))  # Determine grid size

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axs = axs.flatten()
    
    for i, ax in enumerate(axs):
        if i < num_images:
            ax.imshow(images[i].cpu().detach().numpy(), cmap='gray')  # Show image
            ax.axis('off')  # Hide axes
        else:
            ax.remove()  # Remove empty subplots

    plt.tight_layout()
    plt.show()

# Function to generate images from the trained Generator
def generate_images(generator, latent_dim, num_images=10):
    # Sample random latent vectors z from a normal distribution
    z = torch.randn(num_images, latent_dim).to(device)  # Generate random latent vectors
    
    # Generate fake images using the Generator
    generated_images = generator(z)
    
    return generated_images

# After the training loop
print(f'Finished training! Now generating images...')

# Generate images after training
generated_images = generate_images(generator, latent_dim=10, num_images=10)

# Visualize the generated images
show_generated_images(generated_images, num_images=10)

