import os
import shutil
import torchvision
from streaming import MDSWriter
from tqdm import tqdm


def main():
    print("Downloading and converting MNIST dataset to MosaicML streaming format...")

    datasets_dir = "./data/torchdata_mnist"
    streaming_dir = "./data/streaming_mnist"

    os.makedirs(datasets_dir, exist_ok=True)
    if os.path.exists(streaming_dir):
        shutil.rmtree(streaming_dir)
    os.makedirs(streaming_dir, exist_ok=True)

    dataset = torchvision.datasets.MNIST(
        datasets_dir,
        train=True,
        download=True,
    )

    columns = {"image": "pil", "label": "int"}

    with MDSWriter(out=streaming_dir, columns=columns) as out:
        for i in tqdm(range(len(dataset)), desc="Converting"):
            image, label = dataset[i]

            sample = {"image": image, "label": int(label)}

            out.write(sample)


if __name__ == "__main__":
    main()
