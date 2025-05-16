import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import lightning as L
import os


# Simple CNN model for MNIST
class MNISTModel(L.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    datasets_dir = "./data/torchdata_mnist"
    os.makedirs(datasets_dir, exist_ok=True)
    dataset = torchvision.datasets.MNIST(
        datasets_dir,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    batch_size = 1000
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    model = MNISTModel()

    trainer = L.Trainer(max_epochs=1000)
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    main()
