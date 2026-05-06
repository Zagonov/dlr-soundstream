from torch import nn


class ResidualUnit(nn.Module):
    """
    Residual unit from SoundStream paper
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        dilation (int): dilation for the first Conv1d layer
    """

    def __init__(self, n_channels, dilation):
        super().__init__()
        pad_size = (7 - 1) * dilation
        self.net = nn.Sequential(
            nn.ELU(),
            nn.ConstantPad1d((pad_size, 0), 0),
            nn.Conv1d(n_channels, n_channels, kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
        )

    def forward(self, x):
        out = self.net(x)
        return x + out
