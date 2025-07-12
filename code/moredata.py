import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
import timm
import os
import shutil
import re
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from torch.cuda.amp import autocast, GradScaler

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# ... [This section is correct and unchanged] ...
BASE_DATA_PATH = "/home/raj_99/Downloads/Dataset for Crop Pest and Disease Detection"
OLD_BEST_MODEL_PATH = "best_crop_disease_model.pth" 
COMBINED_DATA_DIR = "./combined_organized_dataset"
MODEL_NAME = 'tf_efficientnetv2_s'
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS_HEAD_ONLY = 5
EPOCHS_FULL_TUNE = 15
TOTAL_EPOCHS = EPOCHS_HEAD_ONLY + EPOCHS_FULL_TUNE
LR_HEAD_ONLY = 1e-3
LR_FULL_TUNE = 1e-5
EPOCH_PLOT_DIR = "epoch_plots_expanded"
CHECKPOINT_PATH = "latest_checkpoint_expanded.pth"
BEST_MODEL_PATH = "best_crop_disease_model_expanded.pth"
print(f"Using device: {DEVICE}")

# ==============================================================================
# 2. HELPER FUNCTIONS & DATA PREPARATION
# ==============================================================================
# ... [This section is correct and unchanged] ...
def normalize_class_name(name):
    name = name.lower()
    name = name.replace('___', '_').replace('__', '_').replace(' ', '_')
    name = re.sub(r'\(.*\)', '', name)
    name = name.replace('corn_(maize)', 'maize').replace('pepper,_bell', 'pepper_bell')
    name = re.sub(r'\d+$', '', name)
    return name.strip('_').strip()

def clean_ccmt_name(name):
    return re.sub(r'\d+$', '', name).strip()

def is_valid_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)

def save_epoch_confusion_matrix(epoch, total_epochs, true_labels, predictions, class_names, output_dir):
    labels_for_cm = range(len(class_names))
    cm = confusion_matrix(true_labels, predictions, labels=labels_for_cm)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(max(20, len(class_names)//2.5), max(18, len(class_names)//2.5)))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Validation (Epoch {epoch + 1}/{total_epochs})', fontsize=16)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'cm_epoch_{epoch + 1:02d}.png')
    plt.savefig(plot_path)
    plt.close()

def prepare_combined_dataset(structured_specs, unstructured_specs, target_dir):
    if os.path.exists(target_dir):
        print(f"Target directory '{target_dir}' already exists. Skipping dataset creation.")
        return sorted(os.listdir(os.path.join(target_dir, 'train')))
    print(f"Creating fresh combined dataset in '{target_dir}'...")
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'valid'), exist_ok=True)
    final_class_map = {} 
    for spec in structured_specs:
        print(f"\n--- Processing structured dataset: {spec['name']} ---")
        for split_name in ['train', 'valid']:
            source_split_dir = os.path.join(spec['path'], spec[f'{split_name}_dir'])
            if not os.path.exists(source_split_dir): continue
            for class_name_raw in os.listdir(source_split_dir):
                source_class_path = os.path.join(source_split_dir, class_name_raw)
                if not os.path.isdir(source_class_path): continue
                class_name = f"{spec.get('crop_type', '')}_{clean_ccmt_name(class_name_raw)}" if spec['name'] == 'CCMT' else class_name_raw
                normalized_name = normalize_class_name(class_name)
                if normalized_name not in final_class_map:
                    if split_name == 'train':
                        final_class_map[normalized_name] = class_name
                        print(f"  -> Adding new class: {class_name} (as {normalized_name})")
                    else: continue
                target_class_name = final_class_map[normalized_name]
                target_class_dir = os.path.join(target_dir, split_name, target_class_name)
                os.makedirs(target_class_dir, exist_ok=True)
                for filename in os.listdir(source_class_path):
                    if is_valid_image_file(filename):
                        source_file = os.path.abspath(os.path.join(source_class_path, filename))
                        dest_file = os.path.abspath(os.path.join(target_class_dir, filename))
                        if not os.path.exists(dest_file): os.symlink(source_file, dest_file)
    for spec in unstructured_specs:
        print(f"\n--- Processing unstructured dataset: {spec['name']} ---")
        for class_name_raw in os.listdir(spec['path']):
            source_class_path = os.path.join(spec['path'], class_name_raw)
            if not os.path.isdir(source_class_path): continue
            class_name = f"{spec['name']}_{class_name_raw}"
            normalized_name = normalize_class_name(class_name)
            if normalized_name not in final_class_map:
                final_class_map[normalized_name] = class_name
                print(f"  -> Adding new class: {class_name}")
            target_class_name = final_class_map[normalized_name]
            target_train_dir = os.path.join(target_dir, 'train', target_class_name)
            target_valid_dir = os.path.join(target_dir, 'valid', target_class_name)
            os.makedirs(target_train_dir, exist_ok=True)
            os.makedirs(target_valid_dir, exist_ok=True)
            images = sorted([f for f in os.listdir(source_class_path) if is_valid_image_file(f)])
            split_idx = int(len(images) * 0.8)
            for i, filename in enumerate(images):
                source_file = os.path.abspath(os.path.join(source_class_path, filename))
                target_dir_split = target_train_dir if i < split_idx else target_valid_dir
                dest_file = os.path.abspath(os.path.join(target_dir_split, filename))
                if not os.path.exists(dest_file): os.symlink(source_file, dest_file)
    print("\nDataset aggregation complete.")
    return sorted(list(final_class_map.values()))

STRUCTURED_SPECS = [{'name': 'CCMT', 'crop_type': 'Cashew', 'path': os.path.join(BASE_DATA_PATH, 'CCMT Dataset-Augmented/Cashew'), 'train_dir': 'train_set', 'valid_dir': 'test_set'}, {'name': 'CCMT', 'crop_type': 'Cassava', 'path': os.path.join(BASE_DATA_PATH, 'CCMT Dataset-Augmented/Cassava'), 'train_dir': 'train_set', 'valid_dir': 'test_set'}, {'name': 'CCMT', 'crop_type': 'Maize', 'path': os.path.join(BASE_DATA_PATH, 'CCMT Dataset-Augmented/Maize'), 'train_dir': 'train_set', 'valid_dir': 'test_set'}, {'name': 'CCMT', 'crop_type': 'Tomato', 'path': os.path.join(BASE_DATA_PATH, 'CCMT Dataset-Augmented/Tomato'), 'train_dir': 'train_set', 'valid_dir': 'test_set'}, {'name': 'PlantVillage', 'path': os.path.join(BASE_DATA_PATH, 'archive/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)'), 'train_dir': 'train', 'valid_dir': 'valid'},]
UNSTRUCTURED_SPECS = [{'name': 'Rice', 'path': os.path.join(BASE_DATA_PATH, 'Rice Leaf Disease Images')},]
final_class_names = prepare_combined_dataset(STRUCTURED_SPECS, UNSTRUCTURED_SPECS, COMBINED_DATA_DIR)
NEW_NUM_CLASSES = len(final_class_names)
print(f"\nTotal unique classes after de-duplication: {NEW_NUM_CLASSES}")

# ==============================================================================
# 3. MODEL SURGERY & LOADING
# ==============================================================================
# ... [This section is correct and unchanged] ...
if not os.path.exists(OLD_BEST_MODEL_PATH):
    raise FileNotFoundError(f"Error: Previous model '{OLD_BEST_MODEL_PATH}' not found.")
print("Performing model surgery...")
old_num_classes = 22
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=old_num_classes)
model.load_state_dict(torch.load(OLD_BEST_MODEL_PATH, map_location=DEVICE))
print("Successfully loaded weights from your previously trained model.")
model.reset_classifier(num_classes=NEW_NUM_CLASSES)
print(f"Replaced classifier head. Model now configured for {NEW_NUM_CLASSES} classes.")
model.to(DEVICE)

# ==============================================================================
# 4. DATA LOADING AND TRANSFORMS
# ==============================================================================
# ... [This section is correct and unchanged] ...
os.makedirs(EPOCH_PLOT_DIR, exist_ok=True)
IMG_SIZE = model.default_cfg['input_size'][-1]
train_transforms = transforms.Compose([transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
val_transforms = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = datasets.ImageFolder(os.path.join(COMBINED_DATA_DIR, 'train'), transform=train_transforms, is_valid_file=is_valid_image_file)
valid_dataset = datasets.ImageFolder(os.path.join(COMBINED_DATA_DIR, 'valid'), transform=val_transforms, is_valid_file=is_valid_image_file)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ==============================================================================
# 5. TRAINING SETUP & RESUME LOGIC
# ==============================================================================
# ... [This section is correct and unchanged] ...
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_accuracy = 0.0
start_epoch = 0
optimizer = None
scheduler = None
if os.path.exists(CHECKPOINT_PATH):
    print(f"--- Resuming from checkpoint: {CHECKPOINT_PATH} ---")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_accuracy = checkpoint['best_val_accuracy']
    history = checkpoint['history']
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    if start_epoch < EPOCHS_HEAD_ONLY:
        print("Resuming into Phase 1 (Head Only).")
        for param in model.parameters(): param.requires_grad = False
        for param in model.classifier.parameters(): param.requires_grad = True
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD_ONLY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD_ONLY)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("Resuming into Phase 2 (Full Tune).")
        for param in model.parameters(): param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=LR_FULL_TUNE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL_TUNE)
        if checkpoint['epoch'] >= EPOCHS_HEAD_ONLY:
            print("Checkpoint is from Phase 2, loading optimizer state.")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Checkpoint is from Phase 1. Starting Phase 2 with a fresh optimizer.")
    print(f"--- Resumed training. Starting at Epoch {start_epoch + 1}. Best accuracy so far: {best_val_accuracy:.4f} ---")

# ==============================================================================
# 6. MASTER TRAINING LOOP
# ==============================================================================
for epoch in range(start_epoch, TOTAL_EPOCHS):
    if optimizer is None:
        print("\n--- Starting from scratch. Configuring PHASE 1: Training only the new classifier head ---")
        for param in model.parameters(): param.requires_grad = False
        for param in model.classifier.parameters(): param.requires_grad = True
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD_ONLY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD_ONLY)

    if epoch == EPOCHS_HEAD_ONLY:
        print("\n--- Transitioning to PHASE 2: Fine-tuning all layers ---")
        for param in model.parameters(): param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=LR_FULL_TUNE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL_TUNE)
        print(f"Optimizer reconfigured with learning rate: {LR_FULL_TUNE}")

    phase = "Head Only" if epoch < EPOCHS_HEAD_ONLY else "Full Tune"
    
    model.train()
    running_loss, correct_predictions, total_predictions = 0.0, 0, 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train - {phase}]")
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
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
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_predictions / total_predictions
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)

    model.eval()
    val_loss, correct_predictions, total_predictions = 0.0, 0, 0
    val_all_preds, val_all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Validate]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            val_all_preds.extend(predicted.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())
    val_epoch_loss = val_loss / len(valid_dataset)
    val_epoch_acc = correct_predictions / total_predictions
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)
    print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} -> Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

    # <<< THIS IS THE FIX. Using the correct variables. >>>
    save_epoch_confusion_matrix(epoch, TOTAL_EPOCHS, val_all_labels, val_all_preds, final_class_names, EPOCH_PLOT_DIR)
    print(f"   -> Saved validation confusion matrix for epoch {epoch+1}")

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

print("\n\n--- Expanded Training Finished ---\n")