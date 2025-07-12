import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import time
import argparse
import os

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# --- Configuration ---
MODEL_PATH = "best_crop_disease_model_expanded.pth"
CLASS_NAMES_FILE = "class_names.txt"
MODEL_NAME = 'tf_efficientnetv2_s'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Confidence Threshold ---
# Any prediction with a confidence score below this percentage will be rejected.
# You can tune this value based on your testing. 50.0 is a reasonable start.
CONFIDENCE_THRESHOLD = 50.0

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def load_class_names(filepath):
    """Loads class names from a text file."""
    try:
        with open(filepath, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        print(f"FATAL ERROR: {filepath} not found. Please run save_class_names.py first.")
        exit()

def load_model(model_path, num_classes):
    """Loads the trained model and prepares it for inference."""
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {model_path}.")
        exit()

# ==============================================================================
# 3. MAIN PREDICTION LOGIC
# ==============================================================================
def predict(image_path, model, class_names):
    """
    Takes an image path, performs inference, and returns a result dictionary.
    """
    # 1. Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return {"error": f"Image file not found at {image_path}"}
    
    IMG_SIZE = model.default_cfg['input_size'][-1]
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 2. Perform inference and time it
    with torch.no_grad():
        start_time = time.perf_counter() # Use perf_counter for more precise timing
        outputs = model(image_tensor)
        end_time = time.perf_counter()
    
    inference_time_ms = (end_time - start_time) * 1000

    # 3. Get probabilities and top prediction
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    
    confidence_score = confidence.item() * 100
    predicted_class_name = class_names[predicted_idx.item()]
    
    # 4. Parse the class name into plant and disease
    parts = predicted_class_name.split('_', 1)
    plant_name = parts[0]
    disease_name = parts[1].replace('_', ' ') if len(parts) > 1 else "healthy"
    
    # 5. Apply confidence threshold
    if confidence_score < CONFIDENCE_THRESHOLD:
        is_known = False
        # If unknown, overwrite the plant/disease name for a clearer output
        plant_name_out = "Unknown"
        disease_name_out = "Could not identify with high confidence"
    else:
        is_known = True
        plant_name_out = plant_name
        disease_name_out = disease_name

    return {
        "is_known": is_known,
        "plant_name": plant_name_out,
        "disease_name": disease_name_out,
        "confidence": confidence_score,
        "inference_time_ms": inference_time_ms,
        "best_guess_if_unknown": predicted_class_name # Include the model's best guess for debugging
    }

# ==============================================================================
# 4. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a crop disease from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    # Load everything once
    print("Loading model and class names...")
    class_names_list = load_class_names(CLASS_NAMES_FILE)
    model_loaded = load_model(MODEL_PATH, len(class_names_list))
    print("Model loaded successfully.\n")
    
    # Make the prediction
    result = predict(args.image_path, model_loaded, class_names_list)
    
    # Print the result in a clean, formatted way
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print("--- Prediction Result ---")
        print(f"{'Plant:':<20} {result['plant_name']}")
        print(f"{'Disease/Condition:':<20} {result['disease_name']}")
        print(f"{'Confidence:':<20} {result['confidence']:.2f}%")
        print(f"{'Inference Time:':<20} {result['inference_time_ms']:.2f} ms")
        if not result['is_known']:
            print(f"{'(Model\'s best guess:':<20} {result['best_guess_if_unknown']})")
        print("-------------------------\n")