from Classifier.classifier import classifier

from SAM.SAM import SAM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from typing import Optional, Callable

from torchvision.datasets import GTSRB
from torchvision.transforms import v2

# You may add aditional augmentations, but don't change the output size
_resize_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((32, 32))
])


def get_data_split(transform: Optional[Callable] = _resize_transform):
    """
    Downloads and returns the test and train set of the German Traffic Sign Recognition Benchmark (GTSRB)
    dataset.

    :param transform: An optional transform applied to the images
    :returns: Train and test Dataset instance
    """
    train_set = GTSRB(root="./data", split="train", download=True, transform=transform)
    test_set = GTSRB(root="./data", split="test", download=True, transform=transform)
    return train_set, test_set

def load_model():
    pretrained_weights = torch.load("weights/encoder_weights.pth")

    model_dict = classifier(input_dim=3, num_classes=43).state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict}
    model_dict.update(pretrained_dict)


    model = classifier(input_dim=3, num_classes=43)
    model.load_state_dict(model_dict)

    return model

def train_model_sam(lr, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SAM(model.parameters(), base_optimizer_cls=torch.optim.Adam, lr=lr)

    train_split, _ = get_data_split()
    train_loader = DataLoader(train_split, batch_size=32, shuffle=True)

    batch_loss_history = []
    batch_accuracy_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        model.train()
        for image, label in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            image, label = image.to(device), label.to(device)

            def closure():
                optimizer.zero_grad()
                outputs = model(image)
                loss = criterion(outputs, label)
                loss.backward()
                return loss

            closure()
            optimizer.first_step()

            closure()
            optimizer.second_step()

            outputs = model(image)
            loss = criterion(outputs, label)
            epoch_loss += loss.item()
            batch_loss_history.append(loss.item())  # Save batch loss

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

            # Save batch accuracy
            batch_accuracy = 100 * (predicted == label).sum().item() / label.size(0)
            batch_accuracy_history.append(batch_accuracy)

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        torch.save(model, "weights/classifier_weights_sam.pth")

    metrics = {
        "batch_loss": batch_loss_history,
        "batch_accuracy": batch_accuracy_history,
    }
    torch.save(metrics, "history/training_metrics_sam.pth")


def train_model_normal(lr, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_split, _ = get_data_split()
    train_loader = DataLoader(train_split, batch_size=32, shuffle=True)

    # Initialize lists to store batch-wise loss and accuracy
    batch_loss_history = []
    batch_accuracy_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        model.train()
        for image, label in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(image)

            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_loss_history.append(loss.item())  # Save batch loss

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

            # Save batch accuracy
            batch_accuracy = 100 * (predicted == label).sum().item() / label.size(0)
            batch_accuracy_history.append(batch_accuracy)

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
    
    torch.save(model, "weights/classifier_weights_normal.pth")

    # Optionally, save metrics for visualization
    metrics = {
        "batch_loss": batch_loss_history,
        "batch_accuracy": batch_accuracy_history,
    }
    torch.save(metrics, "history/training_metrics_normal.pth")
    

def eval_model(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(f"weights/classifier_weights_{mode}.pth")
    model = model.to(device)

    _, test_split = get_data_split()

    test_loader = DataLoader(test_split, batch_size=32, shuffle=False)

    model.eval() 

    correct = 0
    total = 0
    test_loss = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  
        for image, label in tqdm(test_loader, desc="Evaluating", unit="batch"):
            image, label = image.to(device), label.to(device)

            outputs = model(image)
            loss = criterion(outputs, label)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  
            correct += (predicted == label).sum().item()
            total += label.size(0)

    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")


