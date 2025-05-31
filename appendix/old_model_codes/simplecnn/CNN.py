import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import os
from pathlib import Path

if __name__ == '__main__':

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")


    data_dir = '/Users/homeyellow/Desktop/2025 S1/DATA3888/Group/images/50'

    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist. Please check the path!")

    subfolders = [f.name for f in Path(data_dir).iterdir() if f.is_dir()]
    if len(subfolders) == 0:
        raise ValueError(f"No class folders found under {data_dir}. Check if images are organized properly!")
    print(f"Found class folders: {subfolders}")

    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    classes = dataset.classes
    num_classes = len(classes)
    print(f"Classes: {classes}")


    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    print(f"Train set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")


    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 12 * 12, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.25)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 12 * 12)
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    model = SimpleCNN(num_classes=num_classes).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 20

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Accuracy: {train_acc:.2f}%")


    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    print("\nValidation Results:")
    print(classification_report(val_labels, val_preds, target_names=classes))


    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    print("\nTest Results:")
    print(classification_report(test_labels, test_preds, target_names=classes))


    if not os.path.exists('results'):
        os.makedirs('results')


    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.savefig('results/training_loss.png')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_accuracies, marker='o')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('results/training_accuracy.png')
    plt.close()

    print("Training curves and confusion matrix saved under 'results/' folder.")


    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    torch.save(model.state_dict(), 'saved_models/simple_cnn.pth')
    print("Model saved to saved_models/simple_cnn.pth")
