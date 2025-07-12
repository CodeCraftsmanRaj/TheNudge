import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
import os
import shutil
import re
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# =======================
# --- CHANGE 1: Import amp for Mixed Precision ---
from torch.cuda.amp import autocast, GradScaler
# =======================

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# --- IMPORTANT: SET THIS TO YOUR DATASET PATH ---
DATASET_ROOT = "/home/raj_99/Downloads/Dataset for Crop Pest and Disease Detection/CCMT Dataset-Augmented"
ORGANIZED_DATA_DIR = "./organized_dataset" 

# --- Model & Training Hyperparameters ---
MODEL_NAME = 'tf_efficientnetv2_s' 
# =======================
# --- CHANGE 2: Reduced Batch Size ---
BATCH_SIZE = 16  # Reduced from 32 to fit in memory
# =======================
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Using model: {MODEL_NAME}")

# ==============================================================================
# 2. DATASET PREPARATION (Automatic Reorganization)
# ==============================================================================

def clean_class_name(name):
    """Removes trailing numbers and extra spaces from class names."""
    return re.sub(r'\d+$', '', name).strip()

def prepare_dataset_structure(root_dir, target_dir):
    """
    Reorganizes the dataset into a standard train/test structure using symlinks.
    """
    if os.path.exists(target_dir):
        print(f"Target directory '{target_dir}' already exists. Skipping reorganization.")
        return
    
    print(f"Organizing dataset into '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)
    
    for split in ['train_set', 'test_set']:
        target_split_dir = os.path.join(target_dir, 'train' if 'train' in split else 'test')
        os.makedirs(target_split_dir, exist_ok=True)
        
        for crop_type in os.listdir(root_dir):
            crop_path = os.path.join(root_dir, crop_type)
            if not os.path.isdir(crop_path): continue
            
            source_split_dir = os.path.join(crop_path, split)
            if not os.path.isdir(source_split_dir): continue
            
            for class_name_raw in os.listdir(source_split_dir):
                class_name_clean = clean_class_name(class_name_raw)
                unique_class_name = f"{crop_type}_{class_name_clean}"
                
                target_class_dir = os.path.join(target_split_dir, unique_class_name)
                os.makedirs(target_class_dir, exist_ok=True)
                
                source_class_dir = os.path.join(source_split_dir, class_name_raw)
                
                for filename in os.listdir(source_class_dir):
                    source_file = os.path.abspath(os.path.join(source_class_dir, filename))
                    dest_file = os.path.abspath(os.path.join(target_class_dir, filename))
                    if not os.path.exists(dest_file):
                       os.symlink(source_file, dest_file)

    print("Dataset reorganization complete.")

prepare_dataset_structure(DATASET_ROOT, ORGANIZED_DATA_DIR)

# ==============================================================================
# 3. DATA LOADING AND TRANSFORMS
# ==============================================================================

model_info = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0) # Get info without classifier
IMG_SIZE = model_info.default_cfg['input_size'][-1]
print(f"Model expects input size: {IMG_SIZE}x{IMG_SIZE}")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_train_dataset = datasets.ImageFolder(os.path.join(ORGANIZED_DATA_DIR, 'train'), transform=train_transforms)
full_test_dataset = datasets.ImageFolder(os.path.join(ORGANIZED_DATA_DIR, 'test'), transform=val_test_transforms)

test_size = len(full_test_dataset)
val_size = test_size // 2
test_size = test_size - val_size
val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])

# Using num_workers=2 is often safer on systems with less RAM
train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

class_names = full_train_dataset.classes
NUM_CLASSES = len(class_names)
print(f"Found {NUM_CLASSES} classes.")

# ==============================================================================
# 4. MODEL, OPTIMIZER, AND LOSS FUNCTION
# ==============================================================================

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# =======================
# --- CHANGE 3: Initialize GradScaler for AMP ---
scaler = GradScaler()
# =======================

# ==============================================================================
# 5. TRAINING AND VALIDATION LOOP
# ==============================================================================

best_val_accuracy = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
BEST_MODEL_PATH = "best_crop_disease_model.pth"

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # =======================
        # --- CHANGE 4: Use autocast for the forward pass ---
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        # =======================

        # =======================
        # --- CHANGE 5: Scale the loss and step the optimizer ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # =======================
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        train_pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(full_train_dataset)
    epoch_acc = correct_predictions / total_predictions
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)

    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with autocast(): # Use autocast in validation too for consistency and speed
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_acc = correct_predictions / total_predictions
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} -> "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

    if val_epoch_acc > best_val_accuracy:
        best_val_accuracy = val_epoch_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"🎉 New best model saved with accuracy: {best_val_accuracy:.4f}")
        
    scheduler.step()

print("Training finished.")

# ==============================================================================
# 6. PLOT TRAINING HISTORY (Same as before)
# ==============================================================================
# (This part of the code remains unchanged)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_plots.png")
plt.show()

# ==============================================================================
# 7. FINAL EVALUATION ON THE TEST SET (Same as before)
# ==============================================================================
# (This part of the code remains unchanged)
print("\n--- Evaluating on the Test Set ---")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.to(DEVICE)
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with autocast(): # Use autocast here too
            outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)
with open("classification_report.txt", "w") as f:
    f.write(report)
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(20, 18))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("\nEvaluation complete. Plots and reports saved.")