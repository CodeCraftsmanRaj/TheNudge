import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import sys
import time

# --- Add the directory containing your 'SoilClassifier' definition if needed ---
# If 'SoilClassifier' is in a different file (e.g., 'model_definition.py'),
# you might need to adjust the Python path or import it differently.
# For simplicity, we'll redefine it here, assuming it's not in a separate module.

# ---------------------- Model Definition (Copied from your evaluation script) ----------------------
class SoilClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=11, pretrained=False): # Ensure num_classes matches your trained model
        super(SoilClassifier, self).__init__()

        if model_name == 'resnet50':
            # Load a base ResNet50 model. We don't need pretrained weights here
            # as we will load our own fine-tuned weights.
            base_model = models.resnet50(weights=None)
            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()  # Remove the original classifier head
            self.features = base_model

            # Create a new classifier head matching the one used during training
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),  # Match dropout used during training
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),  # Match dropout used during training
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------- Prediction Function ----------------------
def predict_single_image(model, image_path, class_names, device, transform):
    """
    Loads an image, preprocesses it, and predicts the soil type using the model.

    Args:
        model (nn.Module): The loaded PyTorch model.
        image_path (str): Path to the input image file.
        class_names (list): List of class names in the order used by the model.
        device (torch.device): The device to run inference on (CPU or CUDA).
        transform (transforms.Compose): The transformations to apply to the image.

    Returns:
        dict: A dictionary containing prediction results:
              'predicted_class': The name of the predicted class.
              'confidence': The confidence score (probability) for the predicted class.
              'all_probabilities': A list of tuples (class_name, probability) sorted descending.
              'error': An error message if prediction failed, None otherwise.
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

        # Set model to evaluation mode and perform inference
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0] # Get probabilities for the single image

        inference_time = time.time() - start_time

        # Get the predicted class index and confidence
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class_name = class_names[predicted_idx.item()]

        # Get all class probabilities and sort them
        probs_dict = {class_names[i]: prob.item() for i, prob in enumerate(probabilities)}
        sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)

        return {
            'predicted_class': predicted_class_name,
            'confidence': confidence.item(),
            'all_probabilities': sorted_probs,
            'inference_time_ms': inference_time * 1000,
            'error': None
        }

    except FileNotFoundError:
        return {'error': f"Error: Image file not found at {image_path}"}
    except Exception as e:
        return {'error': f"An error occurred during prediction: {str(e)}"}

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict soil type from a single image.")
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument(
        "--model_path",
        default='/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass/best_resnet50_soil_model.pth',
        help="Path to the trained model checkpoint (.pth file)."
    )
    args = parser.parse_args()

    # --- Configuration ---
    MODEL_PATH = args.model_path
    IMAGE_PATH = args.image_path

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model Checkpoint ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device) # Load to target device
        class_names = checkpoint['class_names']
        num_classes = len(class_names)
        print(f"Model trained for {num_classes} classes: {class_names}")

        # --- Initialize Model ---
        model = SoilClassifier(model_name='resnet50', num_classes=num_classes).to(device)

        # --- Load Model Weights ---
        # Handle potential mismatches or slightly different state_dict keys if needed
        # Basic loading:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- Define Image Transformations (MUST match those used for testing/validation) ---
    prediction_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Perform Prediction ---
    print(f"\nPredicting soil type for image: {IMAGE_PATH}")
    prediction_result = predict_single_image(model, IMAGE_PATH, class_names, device, prediction_transform)

    # --- Display Results ---
    if prediction_result.get('error'):
        print(prediction_result['error'])
    else:
        print("\n--- Prediction Results ---")
        print(f"Predicted Soil Type: {prediction_result['predicted_class']}")
        print(f"Confidence:          {prediction_result['confidence']:.4f} ({prediction_result['confidence']*100:.2f}%)")
        print(f"Inference Time:      {prediction_result['inference_time_ms']:.2f} ms")

        print("\nTop 3 Probabilities:")
        for i, (class_name, prob) in enumerate(prediction_result['all_probabilities'][:3]):
             print(f"  {i+1}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")

        # Optionally print all probabilities
        # print("\nAll Class Probabilities:")
        # for class_name, prob in prediction_result['all_probabilities']:
        #     print(f"  {class_name}: {prob:.4f}")