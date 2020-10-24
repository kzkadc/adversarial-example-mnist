import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.parameter import Parameter

T = torch.Tensor


class Attacker:
    def __init__(self, target: nn.Module, n_steps: int, eps: float, device: torch.device):
        self.target = target
        self.n_steps = n_steps
        self.eps = eps
        self.device = device

    def make_adversarial_example(self, x: T, t: T):
        self.target.eval()

        delta = Parameter(torch.zeros_like(x).to(self.device))
        opt = Adam([delta])
        for _ in range(self.n_steps):
            opt.zero_grad()
            d_clamped = delta.clamp(-self.eps, self.eps)
            y = self.target((x+d_clamped).clamp(0, 1))
            loss = -F.cross_entropy(y, t)

            loss.backward()
            opt.step()

        return (x+delta).clamp(0, 1).detach()
