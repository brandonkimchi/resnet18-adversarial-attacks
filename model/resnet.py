import torch
from torch import nn, optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt

class ResNet18Trainer:
    def __init__(self, data_dir, num_classes=10, batch_size=64, lr=1e-2, epochs=4, save_path="resnet18_animal10.pth"):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_path = save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        if self.device == "cuda":
            print("GPU name:", torch.cuda.get_device_name(0))

        self._setup_data()
        self._setup_model(num_classes, lr)

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def _setup_data(self):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        full_dataset = ImageFolder(self.data_dir, transform=train_transform)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_set, self.val_set = random_split(
            full_dataset, [train_size, val_size]
        )

        self.val_set.dataset.transform = val_transform

        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False
        )

    def _setup_model(self, num_classes, lr):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        self.model = self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        acc_tot = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            train_accs = []

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                train_accs.append(
                    (outputs.argmax(dim=1) == y).float().mean()
                )

            train_loss = torch.tensor(train_losses).mean()
            train_acc = torch.tensor(train_accs).mean()

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            print(
                f"Epoch {epoch + 1}, "
                f"training loss: {train_loss:.4f}, "
                f"training accuracy: {train_acc:.4f},"
            )

            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            acc_tot[epoch] = val_acc

        self._save_weights()
        self.plot()
        return acc_tot

    def validate(self, epoch):
        self.model.eval()
        losses = []
        accs = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)

                loss = self.criterion(outputs, y)
                losses.append(loss.item())
                accs.append(
                    (outputs.argmax(dim=1) == y).float().mean()
                )

        val_loss = torch.tensor(losses).mean()
        val_acc = torch.tensor(accs).mean()

        print(
            f"Epoch {epoch + 1}, "
            f"validation loss: {val_loss:.4f}, "
            f"validation accuracy: {val_acc:.4f}"
        )

        return val_loss, val_acc

    def _save_weights(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model weights saved to {self.save_path}")

    def plot(self, save_dir=r"C:\Users\Gebruiker\Desktop\Robotics\M6\Deep Learning\Project\media", filename="training_metrics.png"):
        os.makedirs(save_dir, exist_ok=True)

        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 6))

        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.plot(epochs, self.train_accs, label="Train Accuracy")
        plt.plot(epochs, self.val_accs, label="Validation Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training & Validation Metrics")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()

        print(f"Training plot saved to {save_path}")

