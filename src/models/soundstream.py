from torch import nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class SoundStream(nn.Module):
    """
    SoundStream Neural Audio Codec
    """

    def __init__(self, n_channels, latent_channels, strides):
        super().__init__()
        self.encoder = Encoder(n_channels, latent_channels, strides)
        self.decoder = Decoder(latent_channels, n_channels, strides)

    def forward(self, x):
        latent = self.encoder(x)
        generated = self.decoder(latent)
        return {"generated": generated}
