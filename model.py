from torch import nn


def get_model() -> nn.Module:
    N = 32
    return nn.Sequential(
        nn.Conv2d(1, N, kernel_size=4, stride=2, padding=1, bias=False),  # 14
        nn.BatchNorm2d(N),
        nn.ReLU(inplace=True),
        nn.Conv2d(N, N*2, kernel_size=4, stride=2,
                  padding=1, bias=False),   # 7
        nn.BatchNorm2d(N*2),
        nn.ReLU(inplace=True),
        nn.Conv2d(N*2, N*4, kernel_size=3, stride=1,
                  padding=0, bias=False),  # 5
        nn.BatchNorm2d(N*4),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(5*5*N*4, 1000, bias=False),
        nn.BatchNorm1d(1000),
        nn.ReLU(inplace=True),
        nn.Linear(1000, 10)
    )
