import torch
from torch import nn

from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.rvq import ResidualVectorQuantizer


def match_length(audio, target_length):
    current_length = audio.shape[-1]
    if current_length == target_length:
        return audio

    if current_length > target_length:
        return audio[..., :target_length]

    pad_length = target_length - current_length
    pad = audio[..., -1:].expand(*audio.shape[:-1], pad_length)
    return torch.cat([audio, pad], dim=-1)


class SoundStream(nn.Module):
    """
    SoundStream Neural Audio Codec
    """

    def __init__(
        self,
        n_channels,
        latent_channels,
        strides,
        num_quantizers,
        codebook_size,
        gamma,
        kmeans_n_iters,
        code_threshold,
    ):
        super().__init__()
        self.encoder = Encoder(n_channels, latent_channels, strides)
        self.decoder = Decoder(latent_channels, n_channels, strides)
        self.rvq = ResidualVectorQuantizer(
            num_quantizers,
            latent_channels,
            codebook_size,
            gamma,
            kmeans_n_iters,
            code_threshold,
        )

    def forward(self, audio, **batch):
        latent = self.encoder(audio)
        latent = latent.transpose(1, 2)
        rvq_output = self.rvq(latent)
        quantized = rvq_output["quantized"]
        quantized = quantized.transpose(1, 2)
        generated = self.decoder(quantized)
        generated = match_length(generated, audio.shape[-1])
        return {
            "generated": generated,
            "commitment_loss": rvq_output["commitment_loss"],
            "indices": rvq_output["indices"],
        }
