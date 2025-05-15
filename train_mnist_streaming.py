import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import streaming
import torchvision


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


class MNISTDataset(streaming.StreamingDataset):
    def __init__(self, local, batch_size):
        super().__init__(local=local, batch_size=batch_size)
        self.transforms = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        obj = super().__getitem__(idx)
        x = obj["image"]
        y = obj["label"]
        return self.transforms(x), y


def main():
    batch_size = 10000

    datasets_dir = "./data/streaming_mnist"
    dataset = MNISTDataset(local=datasets_dir, batch_size=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = int(5e7)
    print(f"Training on {device} for {num_epochs} epochs...")

    for epoch in trange(num_epochs, desc="training"):
        model.train()
        for batch in loader:
            # Get images and labels from the batch
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

    print("Training complete!")


if __name__ == "__main__":
    main()
