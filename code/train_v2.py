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
from torch.cuda.amp import autocast, GradScaler

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DATASET_ROOT = "/home/raj_99/Downloads/Dataset for Crop Pest and Disease Detection/CCMT Dataset-Augmented"
ORGANIZED_DATA_DIR = "./organized_dataset"
MODEL_NAME = 'tf_efficientnetv2_s'
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NEW: Directory for per-epoch plots ---
EPOCH_PLOT_DIR = "epoch_plots"
CHECKPOINT_PATH = "latest_checkpoint.pth"
BEST_MODEL_PATH = "best_crop_disease_model.pth"

print(f"Using device: {DEVICE}")
print(f"Using model: {MODEL_NAME}")

# ==============================================================================
# 2. HELPER FUNCTIONS (Including new plot function)
# ==============================================================================

def clean_class_name(name):
    return re.sub(r'\d+$', '', name).strip()

def prepare_dataset_structure(root_dir, target_dir):
    # This function remains the same as before
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

def save_epoch_confusion_matrix(epoch_num, true_labels, predictions, class_names_list, output_dir):
    """Generates and saves a confusion matrix plot for a given epoch."""
    cm = confusion_matrix(true_labels, predictions)
    cm_df = pd.DataFrame(cm, index=class_names_list, columns=class_names_list)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Validation Set (Epoch {epoch_num + 1})', fontsize=16)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(output_dir, f'cm_epoch_{epoch_num + 1:02d}.png')
    plt.savefig(plot_path)
    plt.close() # Close the figure to free up memory

# ==============================================================================
# 3. DATA PREPARATION AND LOADING
# ==============================================================================
prepare_dataset_structure(DATASET_ROOT, ORGANIZED_DATA_DIR)
os.makedirs(EPOCH_PLOT_DIR, exist_ok=True) # Create directory for epoch plots

# Data transforms and loaders (same as before)
model_info = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
IMG_SIZE = model_info.default_cfg['input_size'][-1]
print(f"Model expects input size: {IMG_SIZE}x{IMG_SIZE}")
train_transforms = transforms.Compose([transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
val_test_transforms = transforms.Compose([transforms.Resize(IMG_SIZE + 32), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
full_train_dataset = datasets.ImageFolder(os.path.join(ORGANIZED_DATA_DIR, 'train'), transform=train_transforms)
full_test_dataset = datasets.ImageFolder(os.path.join(ORGANIZED_DATA_DIR, 'test'), transform=val_test_transforms)
test_size = len(full_test_dataset)
val_size = test_size // 2
test_size = test_size - val_size
val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])
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
scaler = GradScaler()

# ==============================================================================
# 5. RESUME FROM CHECKPOINT (IF AVAILABLE)
# ==============================================================================
start_epoch = 0
best_val_accuracy = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

if os.path.exists(CHECKPOINT_PATH):
    print(f"--- Resuming from checkpoint: {CHECKPOINT_PATH} ---")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_accuracy = checkpoint['best_val_accuracy']
    history = checkpoint['history']
    print(f"--- Resumed from Epoch {start_epoch}. Best accuracy so far: {best_val_accuracy:.4f} ---")
else:
    print("--- No checkpoint found. Starting training from scratch. ---")

# ==============================================================================
# 6. TRAINING AND VALIDATION LOOP
# ==============================================================================
for epoch in range(start_epoch, EPOCHS):
    # --- Training Phase ---
    model.train()
    running_loss, correct_predictions, total_predictions = 0.0, 0, 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        train_pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(full_train_dataset)
    epoch_acc = correct_predictions / total_predictions
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)

    # --- Validation Phase ---
    model.eval()
    val_loss, correct_predictions, total_predictions = 0.0, 0, 0
    val_all_preds, val_all_labels = [], [] # For confusion matrix
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            # Append batch predictions and labels for confusion matrix
            val_all_preds.extend(predicted.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())
    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_acc = correct_predictions / total_predictions
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

    # --- NEW: Save confusion matrix for this epoch ---
    save_epoch_confusion_matrix(epoch, val_all_labels, val_all_preds, class_names, EPOCH_PLOT_DIR)
    print(f"   -> Saved validation confusion matrix for epoch {epoch+1} to '{EPOCH_PLOT_DIR}'")

    # --- Save Best Model & Checkpoint ---
    if val_epoch_acc > best_val_accuracy:
        best_val_accuracy = val_epoch_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"🎉 New best model saved with accuracy: {best_val_accuracy:.4f}")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'history': history
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    
    scheduler.step()

print("\n\n--- Training Finished ---\n")

# ==============================================================================
# 7. FINAL PLOTTING AND EVALUATION
# ==============================================================================
print("\n--- Plotting Overall Training History ---")
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
plt.savefig("training_history_plots.png")
plt.show()

print("\n--- Evaluating on the Test Set with the BEST model ---")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.to(DEVICE)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
            outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print("\n--- Final Classification Report ---")
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)
with open("classification_report.txt", "w") as f:
    f.write(report)
print("\n--- Final Confusion Matrix ---")
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(20, 18))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Final Confusion Matrix on Test Set')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("final_confusion_matrix.png")
plt.show()
print("\nEvaluation complete. All plots and reports saved.")