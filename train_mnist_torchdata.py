import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import trange


# Simple CNN model for MNIST
class MNISTModel(nn.Module):
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


def main():
    datasets_dir = "./data/torchdata_mnist"
    dataset = torchvision.datasets.MNIST(
        datasets_dir,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    batch_size = 10000
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = int(5e6)
    print(f"Training on {device} for {num_epochs} epochs...")

    for epoch in trange(num_epochs, desc="training"):
        model.train()
        for batch in loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
