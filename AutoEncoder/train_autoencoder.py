from AutoEncoder.autoencoder import AutoEncoder

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

def train_auto_encoder(lr, num_epochs):

    train_loader = load_data()

    autoencoder = AutoEncoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr = lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for images, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(autoencoder.encoder.state_dict(), "weights/encoder_weights.pth")

