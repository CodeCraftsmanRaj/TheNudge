import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

# ---------------------- Dataset Definition ----------------------
class SoilDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data.columns = self.data.columns.str.strip()  # Remove whitespace
        self.img_dir = img_dir
        self.transform = transform
        self.label_cols = self.data.columns[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_vector = row[self.label_cols].values.astype('float32')
        label = torch.tensor(label_vector).argmax().item()

        return image, label

# ---------------------- Paths ----------------------
base_path = '/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass'
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'valid')
test_path = os.path.join(base_path, 'test')

# ---------------------- Transforms ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------- Loaders ----------------------
train_loader = DataLoader(
    SoilDataset(os.path.join(train_path, '_classes.csv'), train_path, transform),
    batch_size=32, shuffle=True)

val_loader = DataLoader(
    SoilDataset(os.path.join(val_path, '_classes.csv'), val_path, transform),
    batch_size=32, shuffle=False)

test_loader = DataLoader(
    SoilDataset(os.path.join(test_path, '_classes.csv'), test_path, transform),
    batch_size=32, shuffle=False)

# ---------------------- Simple CNN Model ----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ---------------------- Training Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=11).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------- Training Loop ----------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_acc = 0
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels)

        val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# ---------------------- Save Model ----------------------
torch.save(model.state_dict(), "soil_model.pth")
print("✅ Model saved as 'soil_model.pth'")

# ---------------------- Test Model ----------------------
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu()
        test_preds.extend(preds)
        test_labels.extend(labels)

test_accuracy = accuracy_score(test_labels, test_preds)
print(f"🧪 Test Accuracy: {test_accuracy:.4f}")
