# Annealing Variational Inference on a bimodal Gaussian mixture

This repository contains the code related to the following paper:

# Code structure

### src

The ``src`` directory contains the source code related to the 2 variational families studied in the paper.

For isotropic Gaussian mixtures, the following modules handle implementation and execution:
*  ``isotropic_gaussian.py`` contains the primary implementation of isotropic gaussian mixtures,
*  ``train_bimodal.py`` contains the training loop for the optimization,
*  ``main_bimodal_gaussian.py`` is a wrapper orchestrating the training process by parsing configuration files.

The modules for RealNVP normalizing have the same structure:
*  ``realnvpwithscale.py`` contains the primary implementation of RealNVP architecture,
*  ``train_realnvp.py`` contains the training loop for optimization,
*  ``main_realnvp.py`` is a wrapper orchestrating the training process by parsing configuration files.

Finally, ``utils.py`` implements some useful functions.

### Experiments

Running experiments requires a ``config.json`` file and an output directory. In ``experiments`` we provide examples for the configurations files.

Here are two example prompts:

```bash
PYTHONPATH=. uv run python src/main_bimodal_gaussian.py --config ./experiments/config_bimodal.json --output-dir ./data/bimodal
PYTHONPATH=. uv run python src/main_realnvp.py --config ./experiments/config_realnvp.json --output-dir ./data/realnvp
```

# Notebooks

``preliminary_experiments.ipynb`` shows some examples of training both isotropic Gaussian mixtures and RealNVP for the exponential annealing schedule highlighted in the paper.
