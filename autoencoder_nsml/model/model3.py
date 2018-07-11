import torch
from torch import nn


class CAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 7, 3, 2),
            nn.BatchNorm2d(7),
            nn.ReLU(),
            SeparableConv2d(7, 24, 3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            SeparableConv2d(24, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeparableConv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SeparableConv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            SeparableConv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            SeparableConv2d(384, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SeparableConv2d(512, 512, 3, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SeparableConv2d(512, 512, 3, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SeparableConv2d(512, 512, 3, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1536, 1024)
        )

        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 1536),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.ConvTranspose2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 120, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 96, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 45, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(45, 18, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(18, 9, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 9, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(9, 9, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(9, 9, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(9, 9, kernel_size=2, stride=1)
        )
        self._init_weights()

    def _init_weights(self):

        def init_func(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
        self.apply(init_func)

    def forward(self, x, only_encoding):
        bs = x.size(0)
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        x = self.fc(self.encoder(x).view(bs, -1))

        if not only_encoding:
            x = self.fc2(x).view(bs, 512, 1, 1)
            x = self.decoder(x)

        return x


class CAEBody(nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel):
        super().__init__()
        self.convs = nn.ModuleList([
            conv_inception_body(in_channel, mid_channel, int(out_channel/3), 5),
            conv_inception_body(in_channel, mid_channel, int(out_channel/3), 3),
            nn.Sequential(
                nn.Conv2d(in_channel, int(out_channel/3), 1),
                nn.ReLU()
            )
        ])

    def forward(self, x):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))

        return torch.cat(outputs, dim=1)


def conv_inception_body(in_channel, mid_channel, out_channel, size):
    return nn.Sequential(
        nn.Conv2d(in_channel, mid_channel, 1),
        nn.ReLU(),
        nn.Conv2d(mid_channel, out_channel, size, 1, 2 if size == 5 else 1),
        nn.ReLU()
    )


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
