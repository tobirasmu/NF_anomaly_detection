import torch
import torch.nn as nn


class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, channels, kernel, stride):
        super(encoder_block, self).__init__()
        conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels=in_channels, out_channels=channels,
                      kernel_size=kernel, padding='same'))
        conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels=channels, out_channels=out_channels,
                      kernel_size=kernel, stride=stride, padding=kernel // 2)
        )
        self.block = nn.Sequential(conv1, nn.LeakyReLU(), conv2, nn.LeakyReLU())

    def forward(self, x):
        return self.block(x)


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, channels, kernel, stride, activation=nn.LeakyReLU()):
        super(decoder_block, self).__init__()
        conv1 = nn.utils.parametrizations.weight_norm(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=channels, kernel_size=kernel, stride=stride,
                               padding=1)
        )
        conv2 = nn.utils.parametrizations.weight_norm(
            nn.ConvTranspose1d(in_channels=channels, out_channels=out_channels, kernel_size=kernel, padding=2)
        )
        self.block = nn.Sequential(conv1, activation, conv2, activation)

    def forward(self, x):
        return self.block(x)


class auto_encoder(nn.Module):
    def __init__(self, in_channels, kernel_size, channels, emb_dim, window_size, variational=True):
        super(auto_encoder, self).__init__()
        reduced_size = channels * (window_size // (2 ** 3 * 3))
        self.variational = variational
        self.window_size = window_size
        # Encoder
        enc1 = encoder_block(in_channels=in_channels, out_channels=channels, channels=channels, kernel=kernel_size, stride=2)
        enc2 = encoder_block(in_channels=channels, out_channels=channels, channels=channels, kernel=kernel_size, stride=2)
        enc3 = encoder_block(in_channels=channels, out_channels=channels, channels=channels, kernel=kernel_size, stride=2)
        enc4 = encoder_block(in_channels=channels, out_channels=channels, channels=channels, kernel=kernel_size, stride=3)
        enc_lin = nn.Linear(reduced_size, emb_dim)
        self.encoder = nn.Sequential(enc1, enc2, enc3, enc4, nn.Flatten(), enc_lin, nn.Tanh())

        # Decoder
        dec_lin = nn.Linear(in_features=emb_dim, out_features=reduced_size)
        dec1 = decoder_block(in_channels=channels, out_channels=channels, channels=channels, kernel=kernel_size, stride=3)
        dec2 = decoder_block(in_channels=channels, out_channels=channels, channels=channels, kernel=kernel_size, stride=2)
        dec3 = decoder_block(in_channels=channels, out_channels=channels, channels=channels, kernel=kernel_size, stride=2)
        dec4 = decoder_block(in_channels=channels, out_channels=in_channels, channels=channels, kernel=kernel_size, stride=2,
                             activation=nn.Identity())
        self.decoder = nn.Sequential(dec_lin, nn.Unflatten(-1, (channels, -1)), dec1, dec2, dec3, dec4)

        # Variational part
        if variational:
            self.mu = torch.nn.Linear(emb_dim, emb_dim)
            self.log_var = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        enc = self.encoder(x)
        if self.variational:
            mu = self.mu(enc)
            log_var = self.log_var(enc)
            sigma = torch.exp(0.5 * log_var)
            if self.training:
                z = torch.randn_like(mu) * sigma + mu  # Re-parametrization step
                dec = self.decoder(z)[:, :, :self.window_size]
            else:
                dec = self.decoder(mu)[:, :, :self.window_size]
            return enc, dec, mu, log_var
        dec = self.decoder(enc)[:, :, :self.window_size]
        return enc, dec

