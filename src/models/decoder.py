from torch import nn

from src.models.residual_unit import ResidualUnit


class DecoderBlock(nn.Module):
    """
    Decoder block from SoundStream paper
    Args:
        n_channels (int): number of output channels
        stride (int): stride for the last Conv1d layer
    """

    def __init__(self, n_channels, stride):
        super().__init__()

        self.net = nn.Sequential(
            nn.ELU(),
            # тут уже паддинг не обязательно симметричный
            # так как можем смотреть в будущее при
            # восстановлении (всё уже известно)
            nn.ConvTranspose1d(
                in_channels=2 * n_channels,
                out_channels=n_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=(stride + 1) // 2,
                output_padding=stride % 2,
            ),
            ResidualUnit(n_channels, dilation=1),
            ResidualUnit(n_channels, dilation=3),
            ResidualUnit(n_channels, dilation=9),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Decoder from SoundStream paper
    Args:
        n_channels (int): number of channels,
        strides (list[int]): list of strides for each DecoderBlock
    """

    def __init__(self, in_channels, n_channels, strides):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16 * n_channels, kernel_size=7, padding=3),
            DecoderBlock(n_channels * 8, strides[3]),
            DecoderBlock(n_channels * 4, strides[2]),
            DecoderBlock(n_channels * 2, strides[1]),
            DecoderBlock(n_channels, strides[0]),
            nn.ELU(),
            nn.Conv1d(n_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)
