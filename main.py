from pathlib import Path

import nprch
import jarch
import torch
import datautils

# length, MNIST = datautils.zip_loader(Path("G:") / "Dataset" / "mnist.zip", suffix="jpg")

print(nprch.nn.Module)
print(jarch.nn.Module)
print(torch.nn.Module)

print(nprch.tensor)
print(jarch.tensor)
print(torch.tensor)
