import torch
import torch.nn as nn
import torch.nn.functional as F

class EModel(nn.Module):
    def __init__(self, API_embed_size, hidden_size, latent_size, num_layers):
        super(EModel, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        #self.batch_size = batch_size

        # Encoder LSTM
        self.lstm = nn.LSTM(API_embed_size, hidden_size, num_layers, batch_first=True)
        
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder LSTM
        self.lstm_decoder = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, API_embed_size)
        
    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        
        h_n = h_n[-1, :, :]  
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        
        return mu, logvar

    def decode(self, z, seq_length):
    
        z = z.unsqueeze(1).repeat(1, seq_length, 1)
        
        outputs, _ = self.lstm_decoder(z)
        reconstructed = self.fc_out(outputs)
        
        return reconstructed

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, masks=None):
        # Encoding step
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)  
        
        # Decoding step
        x_recon = self.decode(z, x.size(1))

        if masks is not None:
            x_recon *= masks.unsqueeze(-1)
        
        return x_recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, masks=None):
        if masks is not None:
            recon_x = recon_x * masks.unsqueeze(-1)
            x = x * masks.unsqueeze(-1)
            reconstruction_loss = F.mse_loss(recon_x, x, reduction='none')
            reconstruction_loss = (reconstruction_loss * masks.unsqueeze(-1)).sum() / masks.sum()
        else:
            reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        return reconstruction_loss + kld_loss