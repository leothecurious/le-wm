"""Encoder, GRU predictor and decoder of Sobal et al. (2022), App. A.7."""

import torch
from torch import nn

from toy.dot.data import SIZE, STEPS

DIM = 512
CHANNELS = 64


def conv(cin, cout, kernel, stride=1):
    return [nn.Conv2d(cin, cout, kernel, stride, kernel // 2), nn.ReLU(), nn.BatchNorm2d(cout)]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(*conv(1, 32, 5, 2), *conv(32, 64, 5, 2), *conv(64, CHANNELS, 3), nn.AvgPool2d(2, 2))
        self.side = SIZE // 4 // 2  # two stride-2 convolutions, then the pooling
        self.linear = nn.Linear(CHANNELS * self.side**2, DIM)

    def forward(self, x):
        return self.linear(self.convs(x[:, None]).flatten(1))


class Decoder(nn.Module):
    """The encoder's layers in reverse order, each stride-2 convolution replaced by upsampling and a stride-1 one."""

    def __init__(self):
        super().__init__()
        self.side = SIZE // 4 // 2
        self.linear = nn.Linear(DIM, CHANNELS * self.side**2)
        up = lambda size: nn.Upsample(size=size)
        self.net = nn.Sequential(
            up(SIZE // 4), *conv(CHANNELS, CHANNELS, 3),
            up(SIZE // 2), *conv(CHANNELS, 32, 5),
            up(SIZE), nn.Conv2d(32, 1, 5, 1, 2),
        )

    def forward(self, s):
        return self.net(self.linear(s).view(-1, CHANNELS, self.side, self.side))[:, 0]


class DotModel(nn.Module):
    """States s_1..s_17 come from the encoder; the predictor, started from s_1, unrolls over the actions."""

    def __init__(self, decoder=False, position_head=False):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = nn.GRU(2, DIM, batch_first=True)
        self.decoder = Decoder() if decoder else None
        self.head = nn.Linear(DIM, 2) if position_head else None

    def encode(self, frames):
        """(B, T, H, W) -> (B, T, DIM)."""
        b = frames.shape[0]
        return self.encoder(frames.flatten(0, 1)).view(b, -1, DIM)

    def predict(self, first, actions):
        """Predicted states s~_2..s~_{T+1} (B, STEPS, DIM) from s_1 (B, DIM) and actions (B, STEPS, 2)."""
        out, _ = self.predictor(actions, first[None].contiguous())
        return out

    def decode(self, states):
        """(B, T, DIM) -> (B, T, H, W)."""
        b = states.shape[0]
        return self.decoder(states.flatten(0, 1)).view(b, -1, SIZE, SIZE)

    def rollout(self, frames, actions):
        """States s_1 and s~_2..s~_{T+1} from the first frame and the actions alone (B, STEPS + 1, DIM)."""
        first = self.encoder(frames[:, 0])
        return torch.cat([first[:, None], self.predict(first, actions)], dim=1)
