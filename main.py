from pathlib import Path

import nprch
import jarch
import torch
import datautils

# length, MNIST = datautils.zip_loader(Path("G:") / "Dataset" / "mnist.zip", suffix="jpg")

a = nprch.Tensor([0, 1, 2], name="hoge")
print(a.shape)
