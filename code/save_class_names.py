import os

# --- CONFIGURATION ---
# This must match the directory created by your training script
COMBINED_DATA_DIR = "./combined_organized_dataset"
CLASS_NAMES_FILE = "class_names.txt"

# --- MAIN LOGIC ---
print("Attempting to save class names...")
train_dir = os.path.join(COMBINED_DATA_DIR, 'train')

if not os.path.exists(train_dir):
    print(f"FATAL ERROR: Directory not found at '{train_dir}'")
    print("Please make sure you have run the training script at least once to create the dataset directory.")
else:
    # Get class names from the directory structure and sort them alphabetically.
    # This alphabetical sorting is CRITICAL as it matches how ImageFolder reads them.
    class_names = sorted(os.listdir(train_dir))
    
    with open(CLASS_NAMES_FILE, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
            
    print(f"\nSUCCESS: Successfully saved {len(class_names)} class names to '{CLASS_NAMES_FILE}'")
    print("You can now run your prediction script.")