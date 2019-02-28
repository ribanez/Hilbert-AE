import torch
from torch import nn


class autoencoder(nn.Module):

    def __init__(self, nc, ndf):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(in_channels=nc,
                      out_channels=ndf,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.ReLU(True),

            # state size. (ndf) x 16 x 16
            nn.Conv2d(in_channels=ndf,
                      out_channels=ndf * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),

            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(in_channels=ndf * 2,
                      out_channels=ndf * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),

            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(in_channels=ndf * 4,
                      out_channels=ndf * 8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
        )

        self.decoder = nn.Sequential(
            # state size. (ndf*4) x 4 x 4
            nn.ConvTranspose2d(in_channels=ndf * 8,
                               out_channels=ndf * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),

            # state size. (ndf*2) x 8 x 8
            nn.ConvTranspose2d(in_channels=ndf * 4,
                               out_channels=ndf * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),

            # state size. (ndf) x 16 x 16
            nn.ConvTranspose2d(in_channels=ndf * 2,
                               out_channels=ndf,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),

            # input is (nc) x 32 x 32
            nn.ConvTranspose2d(in_channels=ndf,
                               out_channels=nc,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
