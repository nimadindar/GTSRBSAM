import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Normalize, Resize
from torchvision.datasets import GTSRB
from tqdm import tqdm


class CustomClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_features, 256) 
        self.relu1 = nn.ReLU()                    
        self.dropout1 = nn.Dropout(0.5)           
        self.fc2 = nn.Linear(256, 128)            
        self.relu2 = nn.ReLU()                    
        self.dropout2 = nn.Dropout(0.5)           
        self.fc3 = nn.Linear(128, num_classes)    

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

resize_transform = Compose([
    ToTensor(),
    Resize((32, 32)),
    Normalize(mean=(0.5,), std=(0.5,))  # Normalize images to a 0-centered range
])

def get_data_split(transform=resize_transform):
    train_set = GTSRB(root="./data", split="train", download=True, transform=transform)
    test_set = GTSRB(root="./data", split="test", download=True, transform=transform)
    return train_set, test_set

train_set, test_set = get_data_split()

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = resnet18(weights='IMAGENET1K_V1')

num_classes = 43

input_features = model.fc.in_features
model.fc = CustomClassifier(input_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")


torch.save(model, "weights/resnet18_weights.pth")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
