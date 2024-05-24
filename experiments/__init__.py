import torch
import nest_asyncio

nest_asyncio.apply()

# Determine the best available device for PyTorch operations (Device Agnostic Code)
if torch.cuda.is_available():
    device = 'cuda'  # GPU
elif torch.backends.mps.is_available():
    device = 'mps'  # GPU for MacOS (Metal Programming Framework)
else:
    device = 'cpu'  # CPU