import torch
from torch import nn
from torch.nn.utils import weight_norm


class ResidualUnit2D(nn.Module):
    def __init__(self, n_channels, m, s):
        super().__init__()

        st, sf = s
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                n_channels,
                m * n_channels,
                kernel_size=(sf + 2, st + 2),
                stride=(sf, st),
                padding=(1, 1),
            ),
        )

        self.skip = nn.Conv2d(
            n_channels,
            m * n_channels,
            kernel_size=(1, 1),
            stride=(sf, st),
        )

    def forward(self, audio):
        return self.net(audio) + self.skip(audio)


class STFTDiscriminator(nn.Module):
    """
    STFT Discriminator for GAN optimization
    """

    def __init__(self, in_channels=2, n_channels=32):
        super().__init__()
        self.register_buffer("window", torch.hann_window(1024))

        self.net = nn.ModuleList(
            [
                nn.Conv2d(in_channels, n_channels, kernel_size=(7, 7), padding=3),
                ResidualUnit2D(n_channels, m=2, s=(1, 2)),
                ResidualUnit2D(2 * n_channels, m=2, s=(2, 2)),
                ResidualUnit2D(4 * n_channels, m=1, s=(1, 2)),
                ResidualUnit2D(4 * n_channels, m=2, s=(2, 2)),
                ResidualUnit2D(8 * n_channels, m=1, s=(1, 2)),
                ResidualUnit2D(8 * n_channels, m=2, s=(2, 2)),
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(16 * n_channels, 1, kernel_size=(int(512 / 2**6), 1)),
                ),
            ]
        )

    def forward(self, audio):
        spec = torch.stft(
            audio.squeeze(1),
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=self.window,
            return_complex=True,
        )
        spec = spec[..., :-1, :]
        spec = torch.view_as_real(spec)
        spec = spec.permute(0, 3, 1, 2)

        feature_map = []
        for module in self.net[:-1]:
            spec = module(spec)
            feature_map.append(spec)

        logits = self.net[-1](spec).squeeze(1, 2)
        return {"feature map": feature_map, "logits": logits}


class NormLeakyConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(*args, **kwargs)), nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x)


class WaveBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.ModuleList(
            [
                NormLeakyConv(1, 16, kernel_size=15, stride=1, padding=7),
                NormLeakyConv(16, 64, kernel_size=41, stride=4, groups=4, padding=20),
                NormLeakyConv(64, 256, kernel_size=41, stride=4, groups=16, padding=20),
                NormLeakyConv(
                    256, 1024, kernel_size=41, stride=4, groups=64, padding=20
                ),
                NormLeakyConv(
                    1024, 1024, kernel_size=41, stride=4, groups=256, padding=20
                ),
                NormLeakyConv(1024, 1024, kernel_size=5, stride=1, padding=2),
                weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
            ]
        )

    def forward(self, audio):
        feature_map = []
        for module in self.net[:-1]:
            audio = module(audio)
            feature_map.append(audio)

        logits = self.net[-1](audio).squeeze(1)

        return {"feature map": feature_map, "logits": logits}


class WaveDiscriminator(nn.Module):
    """
    Wave-based discriminator for GAN optimization
    Состоит из 3 блоков
    Возвращает словарь feature map, logits,
    в каждом из которых по 3 feature map/logits
    """

    def __init__(self):
        super().__init__()

        self.down = nn.AvgPool1d(
            kernel_size=4, stride=2, padding=1, count_include_pad=False
        )
        self.blocks = nn.ModuleList([WaveBlock() for _ in range(3)])

    def forward(self, audio):
        feature_maps = []
        logits = []
        for i in range(3):
            if i != 0:
                audio = self.down(audio)
            output = self.blocks[i](audio)
            feature_maps.append(output["feature map"])
            logits.append(output["logits"])

        return {"feature map": feature_maps, "logits": logits}


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_discriminator = STFTDiscriminator()
        self.wave_discriminator = WaveDiscriminator()

    def forward(self, audio, generated, **batch):
        stft_real_out = self.stft_discriminator(audio)
        stft_gen_out = self.stft_discriminator(generated)
        wave_real_out = self.wave_discriminator(audio)
        wave_gen_out = self.wave_discriminator(generated)

        target_feature_maps = [
            stft_real_out["feature map"],
            *wave_real_out["feature map"],
        ]

        target_logits = [
            stft_real_out["logits"],
            *wave_real_out["logits"],
        ]

        generated_feature_maps = [
            stft_gen_out["feature map"],
            *wave_gen_out["feature map"],
        ]

        generated_logits = [
            stft_gen_out["logits"],
            *wave_gen_out["logits"],
        ]

        return {
            "target_feature_maps": target_feature_maps,
            "generated_feature_maps": generated_feature_maps,
            "target_logits": target_logits,
            "generated_logits": generated_logits,
        }
