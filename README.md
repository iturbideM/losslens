# losslens

Visualize the loss landscape around a trained PyTorch model, in a notebook.

```python
!pip install git+https://github.com/<user>/losslens

import losslens
surf = losslens.surface(model, val_loader, loss_fn=nn.CrossEntropyLoss())
```

Work in progress: plots, summary and comparison are coming next.

Method: filter-normalized random directions, from Li et al.,
*Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018.
