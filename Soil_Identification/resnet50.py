import os
import time
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler

# ---------------------- Dataset Definition ----------------------
class SoilDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data.columns = self.data.columns.str.strip()  # Remove whitespace
        self.img_dir = img_dir
        self.transform = transform
        self.label_cols = self.data.columns[1:]
        
        # Print dataset information
        print(f"Dataset loaded with {len(self.data)} images and {len(self.label_cols)} classes")
        class_counts = {col: self.data[col].sum() for col in self.label_cols}
        print("Class distribution:", class_counts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        # Handle potential file errors gracefully
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image of the expected size as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

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
# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Validation and test transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# ---------------------- Loaders ----------------------
def create_dataloaders(batch_size=32, num_workers=4):
    train_dataset = SoilDataset(os.path.join(train_path, '_classes.csv'), train_path, train_transform)
    val_dataset = SoilDataset(os.path.join(val_path, '_classes.csv'), val_path, val_transform)
    test_dataset = SoilDataset(os.path.join(test_path, '_classes.csv'), test_path, val_transform)
    
    # Get class names for better reporting
    class_names = train_dataset.label_cols.tolist()
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_names

# ---------------------- Model Definition ----------------------
class SoilClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=11, pretrained=True):
        super(SoilClassifier, self).__init__()
        
        # Load pre-trained model
        if model_name == 'resnet50':
            base_model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()  # Remove classifier
            self.features = base_model
            
            # Create new classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'efficientnet_b3':
            base_model = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()  # Remove classifier
            self.features = base_model
            
            # Create new classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------- Training Functions ----------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backpropagation with scaler for mixed precision
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_acc, all_preds, all_labels

# ---------------------- Main Training Loop ----------------------
def train_model(
    model_name='resnet50', 
    num_epochs=30, 
    batch_size=32,
    lr=0.0003,
    weight_decay=1e-4,
    patience=7  # Early stopping patience
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(batch_size)
    
    # Initialize model
    num_classes = len(class_names)
    model = SoilClassifier(model_name=model_name, num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize mixed precision training
    scaler = GradScaler()
    
    # Training tracking
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Training loop
    start_time = time.time()
    
    print(f"Starting training {model_name} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, f"best_{model_name}_soil_model.pth")
            print(f"✅ Model improved - saved checkpoint at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. "
                  f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
            break
    
    # Print training summary
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    # Load best model for evaluation
    checkpoint = torch.load(f"best_{model_name}_soil_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history, class_names

# ---------------------- Evaluation Functions ----------------------
def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    
    # Get actual classes present in test set predictions and labels
    unique_classes = np.unique(np.concatenate([all_labels, all_preds]))
    present_class_indices = sorted(unique_classes)
    present_class_names = [class_names[i] for i in present_class_indices]
    
    # Generate classification report with only the classes present in the data
    class_report = classification_report(
        all_labels, all_preds, 
        labels=present_class_indices,
        target_names=present_class_names, 
        digits=4,
        zero_division=0
    )
    
    # Generate full confusion matrix with all classes
    # This handles missing classes correctly
    num_classes = len(class_names)
    conf_matrix = confusion_matrix(
        all_labels, all_preds, 
        labels=list(range(num_classes))
    )
    
    print(f"🧪 Test Accuracy: {test_acc:.4f}")
    print("Classification Report:")
    print(class_report)
    
    # Print missing classes if any
    all_class_indices = set(range(len(class_names)))
    missing_indices = all_class_indices - set(present_class_indices)
    if missing_indices:
        missing_classes = [class_names[i] for i in missing_indices]
        print(f"\nNote: The following classes were missing in the test predictions: {missing_classes}")
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix.png")
    plt.close()
    
    return test_acc, class_report, conf_matrix

# ---------------------- Plotting Functions ----------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# ---------------------- Run Training and Evaluation ----------------------
if __name__ == "__main__":
    # Configuration
    config = {
        'model_name': 'resnet50',  # Options: 'resnet50', 'efficientnet_b3'
        'num_epochs': 30,
        'batch_size': 32,
        'lr': 0.0003,
        'weight_decay': 1e-4,
        'patience': 7
    }
    
    # Train model
    model, history, class_names = train_model(**config)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = create_dataloaders(batch_size=config['batch_size'])
    test_acc, class_report, conf_matrix = evaluate_model(model, test_loader, device, class_names)
    
    print(f"\n📊 Final Model Performance:")
    print(f"Model: {config['model_name']}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Results saved to confusion_matrix.png and training_history.png")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'config': config
    }, f"final_{config['model_name']}_soil_model.pth")
    print(f"✅ Final model saved as 'final_{config['model_name']}_soil_model.pth'")


    # default_config = {
    # 'model_name': 'resnet50',  # Options: 'resnet50', 'efficientnet_b3'
    # 'num_epochs': 30,
    # 'batch_size': 32,
    # 'lr': 0.0003,
    # 'weight_decay': 1e-4,
    # 'patience': 7
    # }

    # # Default class names (if not found in checkpoint)
    # default_class_names = ['Alluvial', 'Black', 'Cinder', 'Clay', 'Laterite', 'Loamy', 'Peat', 'Red', 'Sandy', 'Yellow', 'loam']

    # # Define device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Load checkpoint
    # checkpoint = torch.load("/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass/best_resnet50_soil_model.pth", map_location=device)

    # # Load class names and config safely
    # class_names = checkpoint.get('class_names', default_class_names)
    # config = checkpoint.get('config', default_config)

    # # missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)


    # # Initialize the model
    # model = models.resnet50(weights=None)  # Use weights=None as pretrained is deprecated
    # model.fc = nn.Linear(model.fc.in_features, len(class_names))
    # missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    # model.to(device)

    # # Load test data
    # _, _, test_loader, _ = create_dataloaders(batch_size=config['batch_size'])

    # # Evaluate model
    # test_acc, class_report, conf_matrix = evaluate_model(model, test_loader, device, class_names)

    # print(f"\n📊 Final Model Performance:")
    # print(f"Model: {config['model_name']}")
    # print(f"Test Accuracy: {test_acc:.4f}")
    # print(f"Results saved to confusion_matrix.png and training_history.png")

    # # Save final model
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'class_names': class_names,
    #     'config': config
    # }, f"final_{config['model_name']}_soil_model.pth")

    # print(f"✅ Final model saved as 'final_{config['model_name']}_soil_model.pth'")