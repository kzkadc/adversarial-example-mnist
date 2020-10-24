from ignite.engine.engine import Engine
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from pathlib import Path
import copy


class MetricsAccumulator:
    def __init__(self, evaluator: Engine, train_loader: DataLoader, test_loader: DataLoader):
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.epochs = []
        self.train_metrics = []
        self.test_metrics = []

    def __call__(self, engine: Engine):
        self.epochs.append(engine.state.epoch)

        self.evaluator.run(self.train_loader)
        self.train_metrics.append(copy.copy(self.evaluator.state.metrics))

        self.evaluator.run(self.test_loader)
        self.test_metrics.append(copy.copy(self.evaluator.state.metrics))

    def get_metric(self, train: bool, key: str):
        metrics = self.train_metrics if train else self.test_metrics
        return [m[key] for m in metrics]


def plot_metrics(log: MetricsAccumulator,  y_keys: list, out_file: Path):
    def _plot(engine: Engine):
        plt.figure()
        for yk in y_keys:
            plt.plot(log.epochs, log.get_metric(
                train=True, key=yk), label="train_"+yk)
            plt.plot(log.epochs, log.get_metric(
                train=False, key=yk), label="test_"+yk)

        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(str(out_file))
        plt.close()

    return _plot


def print_metrics(log: MetricsAccumulator, y_keys: list):
    def _print(engine: Engine):
        s = (
            f"epoch: {log.epochs[-1]:d}",
            *(f"train_{yk}: {log.get_metric(True,yk)[-1]}" for yk in y_keys),
            *(f"test_{yk}: {log.get_metric(False,yk)[-1]}" for yk in y_keys)
        )
        print(", ".join(s))

    return _print


def save_model(net: nn.Module, save_dir_path: Path):
    try:
        save_dir_path.mkdir(parents=True)
    except FileExistsError:
        pass

    def _save(engine: Engine):
        p = str(save_dir_path / f"model_epoch-{engine.state.epoch:03d}.pt")
        torch.save(net.state_dict(), p)

    return _save
