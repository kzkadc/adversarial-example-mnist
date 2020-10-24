import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from ignite.engine.engine import Engine, Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from pathlib import Path

from model import get_model
from handlers import MetricsAccumulator, plot_metrics, print_metrics, save_model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", default="result", help="result directory")
    parser.add_argument("-e", type=int, default=10, help="epoch")
    parser.add_argument("-b", type=int, default=64, help="batch size")
    parser.add_argument("-g", type=int, default=-1, help="GPU id")

    args = parser.parse_args()
    main(args)


def main(args):
    if torch.cuda.is_available() and args.g >= 0:
        device = torch.device(f"cuda:{args.g:d}")
        print(f"GPU mode: {args.g:d}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    result_dir_path = Path(args.d)
    try:
        result_dir_path.mkdir(parents=True)
    except FileExistsError:
        pass

    train_dataset = MNIST(root=".", train=True,
                          download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root=".", train=False,
                         download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.b)

    net = get_model().to(device)
    opt = Adam(net.parameters())

    trainer = Engine(MNISTTrainer(net, opt, device))

    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(F.cross_entropy)
    }
    evaluator = create_supervised_evaluator(net, metrics)

    logger = MetricsAccumulator(evaluator, train_loader, test_loader)

    trigger = Events.EPOCH_COMPLETED
    trainer.add_event_handler(trigger, logger)
    trainer.add_event_handler(trigger, plot_metrics(
        logger, ["accuracy"], result_dir_path/"accuracy.pdf"))
    trainer.add_event_handler(trigger, plot_metrics(
        logger, ["loss"], result_dir_path/"loss.pdf"))
    trainer.add_event_handler(
        trigger, print_metrics(logger, ["accuracy", "loss"]))
    trainer.add_event_handler(
        trigger, save_model(net, result_dir_path/"models"))

    trainer.run(train_loader, max_epochs=args.e)


class MNISTTrainer:
    def __init__(self, net: nn.Module, opt: Optimizer, device: torch.device):
        self.net = net
        self.opt = opt
        self.device = device

    def __call__(self, engine: Engine, batch: tuple):
        self.net.train()
        self.opt.zero_grad()

        x, t = map(lambda x: x.to(self.device), batch)
        y = self.net(x)
        loss = F.cross_entropy(y, t)
        loss.backward()
        self.opt.step()

        return loss


if __name__ == "__main__":
    parse_args()
