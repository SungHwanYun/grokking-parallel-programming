# Chapter 7 — Work-efficient scan: Blelloch
# "Scan in PyTorch": the same scan as library one-liners in NumPy and PyTorch,
# and the inclusive-to-exclusive conversion (both listings from that section).

import numpy as np
arr = np.array([3, 1, 7, 0, 4, 1, 6, 3])
result = np.cumsum(arr)    # Inclusive scan: [3, 4, 11, 11, 15, 16, 22, 25]
import torch
tensor = torch.tensor([3, 1, 7, 0, 4, 1, 6, 3])
result = torch.cumsum(tensor, dim=0)    # The same inclusive scan as a tensor: tensor([3, 4, 11, 11, 15, 16, 22, 25])

# If you need an exclusive scan, shift the inclusive result right by one
inclusive = np.cumsum(arr)    # NumPy's inclusive scan: [3, 4, 11, 11, 15, 16, 22, 25]
exclusive = np.concatenate(([0], inclusive[:-1]))    # Shift right and pad a zero at the front: [0, 3, 4, 11, 11, 15, 16, 22]
inclusive = torch.cumsum(tensor, dim=0)    # The same inclusive scan, now as a tensor
exclusive = torch.cat([torch.tensor([0]), inclusive[:-1]])    # The same shift: tensor([0, 3, 4, 11, 11, 15, 16, 22])
