import zipfile
from pathlib import Path
import io
from typing import Iterator
from dataclasses import dataclass


@dataclass
class ZipFileByteData:
    path: Path
    data: io.BytesIO


def zip_loader(zip_path: Path, suffix="png", all=False) -> tuple[int, Iterator[ZipFileByteData]]:
    zip_path = zip_path.resolve()
    if not all:
        assert suffix, "suffix must be any charactor. if you want to extract all files from zip, please set all=True."
    if not suffix.startswith("."):
        suffix = "." + suffix

    print("Loading: ", zip_path)
    print("Extract suffix: ", suffix)

    cnt = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for f in zf.infolist():
            if (not all) and Path(f.filename).suffix != suffix:
                continue
            cnt += 1

    def _zip_loader():
        with zipfile.ZipFile(zip_path, "r") as zf:
            for f in zf.infolist():
                rf = Path(f.filename)
                if (not all) and rf.suffix != suffix:
                    continue
                with zf.open(f.filename) as data:
                    yield ZipFileByteData(path=rf.resolve(), data=io.BytesIO(data.read()))
        return None

    return cnt, _zip_loader()


if __name__ == "__main__":
    from tqdm import tqdm

    length, loader = zip_loader(Path("G:") / "Dataset" / "mnist.zip", suffix="jpg")
    for i, file in tqdm(enumerate(loader), total=length):
        if not i % 10000:
            tqdm.write(f"{i}, {file}")
