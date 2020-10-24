import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import numpy as np
import cv2

from pathlib import Path

from adversarial_attacker import Attacker
from model import get_model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", required=True, help="target model")
    parser.add_argument("--steps", type=int, default=1000,
                        help="number of steps for attack")
    parser.add_argument("--eps", type=float, default=0.1,
                        help="upper limit of infinity norm of adversarial perturbation")
    parser.add_argument("-n", type=int, default=5,
                        help="number of adversarial examples to generate")
    parser.add_argument("-g", type=int, default=-1, help="GPU id")
    parser.add_argument("-d", required=True,
                        help="directory to save output images")

    args = parser.parse_args()
    main(args)


def main(args):
    if torch.cuda.is_available() and args.g >= 0:
        device = torch.device(f"cuda:{args.g:d}")
        print(f"GPU mode: {args.g:d}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    test_dataset = MNIST(root=".", train=False,
                         download=True, transform=ToTensor())
    indices = np.random.choice(len(test_dataset), size=args.n, replace=False)
    x = torch.stack([test_dataset[i][0] for i in indices]).to(device)
    t = torch.tensor([test_dataset[i][1] for i in indices]).to(device)

    net = get_model().to(device)
    net.load_state_dict(torch.load(args.m))

    attacker = Attacker(net, args.steps, args.eps,  device)
    perturbed = attacker.make_adversarial_example(x, t)

    with torch.no_grad():
        y_clean = net(x)
        y_perturbed = net(perturbed)
        clean_prob, clean_pred = map(lambda x: x.cpu().numpy(),
                                     F.softmax(y_clean, dim=1).max(dim=1))
        clean_loss = F.cross_entropy(
            y_clean, t, reduction="none").cpu().numpy()
        attacked_prob, attacked_pred = map(lambda x: x.cpu().numpy(),
                                           F.softmax(y_perturbed, dim=1).max(dim=1))
        attacked_loss = F.cross_entropy(
            y_perturbed, t, reduction="none").cpu().numpy()

    x = x.cpu().numpy()
    perturbed = perturbed.cpu().numpy()
    out_img = np.concatenate([x, perturbed], axis=3).squeeze()
    out_img = (out_img*255).astype(np.uint8)

    out_dir_path = Path(args.d)
    try:
        out_dir_path.mkdir(parents=True)
    except FileExistsError:
        pass

    for i in range(args.n):
        print(f"prediction for original: {clean_pred[i]} ({clean_prob[i]*100}%, loss={clean_loss[i]})\n"
              f"prediction for attacked: {attacked_pred[i]} ({attacked_prob[i]*100}%, loss={attacked_loss[i]})\n")
        cv2.imwrite(
            str(out_dir_path/f"out_{i:03d}.png"), out_img[i])


if __name__ == "__main__":
    parse_args()
