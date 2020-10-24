# Adversarial example on MNIST
Training MNIST classification model and generating adversarial example.

Szegedy, Christian, et al. "Intriguing properties of neural networks." arXiv preprint arXiv:1312.6199 (2013).

## Requirements
PyTorch, PyTorch-Ignite, OpenCV, Matplotlib

PyTorch: see [the official document](https://pytorch.org/get-started/locally/).

```bash
$ pip install pytorch-ignite opencv-python matplotlib
```

## Usage
### Training Classification Model
Train the MNIST classifier to be attacked with `train.py`.

Trained models (`result_directory/models/*.pt`) are used in the next step.

```
usage: train.py [-h] [-d D] [-e E] [-b B] [-g G]

optional arguments:
  -h, --help  show this help message and exit
  -d D        result directory
  -e E        epoch
  -b B        batch size
  -g G        GPU id
```

You can use `trained_mnist_classifier.pt` instead.

## Generating Adversarial Example
Run `generate_sample.py`.

Model files (`*.pt`) can be passed to `-m` option.

```
usage: generate_sample.py [-h] -m M [--steps STEPS] [--eps EPS] [-n N] [-g G] -d D

optional arguments:
  -h, --help     show this help message and exit
  -m M           target model
  --steps STEPS  number of steps for attack
  --eps EPS      upper limit of infinity norm of adversarial perturbation
  -n N           number of adversarial examples to generate
  -g G           GPU id
  -d D           directory to save output images
```

Images like the following are output.

![output_sample](output_sample.png)

Left: original image; right: adversarial example
