# losslens

See what the loss looks like around your trained PyTorch model.

`losslens` takes a trained model, moves its weights a little in two random directions, and
measures the loss at every point of a grid. The result is a 2D map (and a 3D surface) of the
loss around your model: a wide, flat valley or a narrow, sharp one.

## Notebooks

- **Tour of everything losslens does:**
  [open in Colab](https://colab.research.google.com/github/iturbideM/losslens/blob/main/examples/losslens_tour.ipynb)
- **The paper's pretrained ResNet-56 models (with and without skip connections), compared with
  the paper's figures:**
  [open in Colab](https://colab.research.google.com/github/iturbideM/losslens/blob/main/examples/paper_resnets.ipynb)
- **Homework template:**
  [open in Colab](https://colab.research.google.com/github/iturbideM/losslens/blob/main/examples/homework_template.ipynb)

## Install

```python
!pip install git+https://github.com/iturbideM/losslens
```

## Quick start

```python
import torch.nn as nn
import losslens

surf = losslens.surface(model, val_loader, loss_fn=nn.CrossEntropyLoss())

surf.plot()        # 2D contour map; the red star is your trained model
surf.plot_3d()     # interactive 3D surface: drag to rotate
surf.plot_slice()  # 1D cuts through your model along each direction
surf.summary()     # key numbers about the landscape
```

Compare several models on the same color scale:

```python
losslens.compare({"no skip": surf_a, "with skip": surf_b})
```

Save the 3D plot as a web page that opens in any browser (works offline):

```python
surf.plot_3d().write_html("landscape_3d.html")
```

Save a surface so you don't have to recompute it:

```python
surf.save("resnet.npz")
surf = losslens.load("resnet.npz")
```

## Options

| Option | Default | What it does |
|---|---|---|
| `scale` | `1.0` | How far from your model to look. Bigger shows a wider area. At 1.0, each layer's weights are changed by up to 100% of their own size. |
| `resolution` | `25` | Grid points per side. Higher is sharper and slower (25 means 625 loss evaluations). |
| `n_samples` | `2000` | How many examples the loss is averaged over. |
| `seed` | `0` | Which random directions. Try another seed to check that the shape isn't a fluke. |
| `device` | `"auto"` | Where to compute: your model's device, otherwise the GPU if there is one. |

The plotting functions take `log=True` (show log10 of the loss, useful when it grows by
orders of magnitude) and `zmax=` (top of the color scale).

## Data and loss

`data` can be a DataLoader, a Dataset, or a tuple of tensors `(X, y)`.

If each batch is `(inputs, targets)`, pass `loss_fn`, called as `loss_fn(model(inputs), targets)`.
Use a loss that averages over the batch (the default for PyTorch losses).

For anything else, pass `forward(model, batch) -> loss` instead:

```python
# Autoencoder (no labels)
surf = losslens.surface(ae, loader, forward=lambda m, x: F.mse_loss(m(x), x))

# Model taking a dict batch
surf = losslens.surface(net, loader, forward=lambda m, b: m(**b).loss)
```

## What `summary()` reports

- **loss of the trained model**, and the **lowest loss on the grid** and where it is. If the
  lowest point is not your model, the model is not at a minimum along these directions.
- **mean loss increase at 10%, 25% and 50% of `scale`**: how fast the loss rises as you move
  away from your model. Smaller means a flatter minimum.
- **steeper / flatter direction**: 1.0 means the valley looks the same in both directions.

## Good to know

- Your model is never modified. The loss is computed in eval mode (dropout off, BatchNorm using
  its running statistics).
- Biases, BatchNorm parameters and frozen layers are not moved.
- Because the directions are scaled to each layer's own weights, plots of different models are
  comparable, as long as they use the same `scale` and `n_samples`.

Method: Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018.
