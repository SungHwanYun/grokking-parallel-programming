# Chapter 7 — Work-Efficient Scan: Blelloch
# "Scan in PyTorch": the same scan as library one-liners in NumPy and PyTorch,
# and the inclusive-to-exclusive conversion.

import numpy as np
arr = np.array([3, 1, 7, 0, 4, 1, 6, 3])
result = np.cumsum(arr)
# [3, 4, 11, 11, 15, 16, 22, 25]  ← inclusive scan
import torch
tensor = torch.tensor([3, 1, 7, 0, 4, 1, 6, 3])
result = torch.cumsum(tensor, dim=0)
# tensor([3, 4, 11, 11, 15, 16, 22, 25])

# NumPy: inclusive to exclusive scan
inclusive = np.cumsum(arr)          # [3, 4, 11, 11, 15, 16, 22, 25]
exclusive = np.concatenate(([0], inclusive[:-1]))
# [0, 3, 4, 11, 11, 15, 16, 22]  ← exclusive scan!
# PyTorch: inclusive to exclusive scan
inclusive = torch.cumsum(tensor, dim=0)
exclusive = torch.cat([torch.tensor([0]), inclusive[:-1]])
# tensor([0, 3, 4, 11, 11, 15, 16, 22])
