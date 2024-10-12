import torch
import torch.nn as nn

# VAE Definition
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=False),  # Ensure no in-place modifications
            nn.Linear(256, latent_dim * 2)  # Outputs mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mean, log_var = encoded.chunk(2, dim=-1)
        z = self.reparameterize(mean, log_var)
        decoded = self.decoder(z)
        return decoded, mean, log_var

# GAN Generator Definition
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# GAN Discriminator Definition
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Autoregressive Model Definition
class AutoregressiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(AutoregressiveModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_out)
        return out, hidden

# Reinforcement Learning Agent Definition
class ReinforcementLearningAgent:
    def __init__(self, generator, discriminator, latent_dim, learning_rate=0.001):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

    def get_action(self, state):
        z = torch.randn(state.size(0), self.latent_dim).to(state.device)
        return self.generator(z)

    def update(self, fake_data, rewards):
        self.optimizer.zero_grad()
        fake_data_log_prob = torch.log(fake_data + 1e-8)
        loss = -(rewards * fake_data_log_prob).mean()
        loss.backward()
        self.optimizer.step()
