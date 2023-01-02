from pathlib import Path
from itertools import islice
import matplotlib.pyplot as plt
from PIL import Image

import nprch
import datautils

length, MNIST = datautils.zip_loader(Path("G:") / "Dataset" / "mnist.zip", suffix="jpg")

print(nprch.nn.Module)