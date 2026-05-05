from torch import nn
from residual_unit import ResidualUnit


class EncoderBlock(nn.Module):
    """
    Encoder block from SoundStream paper
    Args:
        n_channels (int): number of output channels
        stride (int): stride for the last Conv1d layer
    """

    def __init__(self, n_channels, stride):
        super().__init__()

        self.net = nn.Sequential(
            ResidualUnit(n_channels // 2, dilation=1),
            ResidualUnit(n_channels // 2, dilation=3),
            ResidualUnit(n_channels // 2, dilation=9),
            nn.ELU(),
            nn.ConstantPad1d((stride, 0), 0),
            nn.Conv1d(n_channels // 2, n_channels, kernel_size=2 * stride, stride=stride)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """
    Encoder from SoundStream paper
    Args:
        n_channels (int): number of channels,
        strides (list[int]): list of strides for each EncoderBlock
    """

    def __init__(self, n_channels, out_channels, strides):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConstantPad1d((7 - 1, 0), 0),
            nn.Conv1d(1, n_channels, kernel_size=7),

            EncoderBlock(n_channels * 2, strides[0]),
            EncoderBlock(n_channels * 4, strides[1]),
            EncoderBlock(n_channels * 8, strides[2]),
            EncoderBlock(n_channels * 16, strides[3]),
            nn.ELU(),

            nn.ConstantPad1d((3 - 1, 0), 0),
            nn.Conv1d(n_channels * 16, out_channels, kernel_size=3)
        )

    def forward(self, x):
        return self.net(x)
