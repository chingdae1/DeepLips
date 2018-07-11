import torch
from torch import nn


class CAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            CAEBody(2, 1, 3),
            nn.Conv2d(3, 5, kernel_size=3, stride=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            CAEBody(5, 3, 9),
            nn.Conv2d(9, 25, kernel_size=3, stride=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            CAEBody(96, 56, 252),
            nn.Conv2d(252, 360, kernel_size=3, stride=2),
            nn.BatchNorm2d(360),
            nn.ReLU(),
            nn.Conv2d(360, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024)
        )

        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 120, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(120, 96, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 45, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(45, 18, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(18, 9, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 5, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(5, 4, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1)
        )
        self._init_weights()

    def _init_weights(self):

        def init_func(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
        self.apply(init_func)

    def forward(self, x, only_encoding):
        bs = x.size(0)
        x = x.unsqueeze(1)
        x = self.fc(self.encoder(x).view(bs, -1))

        if not only_encoding:
            x = self.fc2(x).view(bs, 512, 2, 2)
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
