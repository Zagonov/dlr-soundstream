from math import sqrt

import torch
import torchaudio
from torch import nn


class ReconstructionLoss(nn.Module):
    """
    Multi-scale spectral reconstruction loss
    """

    def __init__(
        self, sample_rate=16000, window_sizes=[64, 128, 256, 512, 1024, 2048], n_mels=64
    ):
        super().__init__()
        self.window_sizes = window_sizes
        self.transforms = nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=window_size,
                win_length=window_size,
                hop_length=window_size // 4,
                n_mels=n_mels,
            )
            for window_size in window_sizes
        )

    def forward(self, generated, audio):
        """
        Calculate multi-scale spectral reconstruction loss.

        Args:
            generated (Tensor): generated audio signal.
            audio (Tensor): target audio signal.

        Returns:
            reconstruction loss (dict).

        """
        generated = generated.squeeze(1)
        audio = audio.squeeze(1)

        loss = 0
        for transform, window_size in zip(self.transforms, self.window_sizes):
            gen_spec = transform(generated)
            target_spec = transform(audio)
            loss += torch.abs(gen_spec - target_spec).mean()
            log_gen_spec = torch.log(gen_spec + 1e-6)
            log_target_spec = torch.log(target_spec + 1e-6)
            alpha = sqrt(window_size / 2)
            loss += (
                alpha
                * torch.linalg.vector_norm(
                    log_gen_spec - log_target_spec, dim=-2
                ).mean()
            )

        loss = loss / len(self.transforms)

        return {"loss": loss}
