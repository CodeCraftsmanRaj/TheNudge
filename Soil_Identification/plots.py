import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import sys
import time
import matplotlib.pyplot as plt # Import matplotlib

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
              'inference_time_ms': Time taken for inference in milliseconds.
              'error': An error message if prediction failed, None otherwise.
    """
    try:
        # Load image (keep original for plotting later if needed)
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

# ---------------------- Plotting Function ----------------------
def plot_prediction_results(image_path, prediction_result, output_path=None):
    """
    Creates a plot showing the input image and a bar chart of prediction probabilities.

    Args:
        image_path (str): Path to the input image file.
        prediction_result (dict): The dictionary returned by predict_single_image.
        output_path (str, optional): Path to save the plot image. If None, displays the plot.
    """
    if prediction_result.get('error'):
        print(f"Cannot plot results due to prediction error: {prediction_result['error']}")
        return

    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not load image {image_path} for plotting.")
        return
    except Exception as e:
        print(f"Error loading image {image_path} for plotting: {e}")
        return

    # Extract data for plotting
    sorted_probs = prediction_result['all_probabilities']
    class_names = [item[0] for item in sorted_probs]
    probabilities = [item[1] for item in sorted_probs]
    predicted_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence']

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # 1 row, 2 columns

    # --- Plot 1: Input Image ---
    axes[0].imshow(img)
    axes[0].set_title(f"Input Image\n(Predicted: {predicted_class} - {confidence*100:.1f}%)")
    axes[0].axis('off') # Hide axes ticks

    # --- Plot 2: Probability Bar Chart ---
    # Use horizontal bars for better readability of class names
    y_pos = range(len(class_names))
    bars = axes[1].barh(y_pos, probabilities, align='center', color='skyblue')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(class_names)
    axes[1].invert_yaxis()  # Highest probability at the top
    axes[1].set_xlabel('Probability')
    axes[1].set_title('Prediction Probabilities')
    axes[1].set_xlim(0, 1) # Probabilities range from 0 to 1

    # Add probability values as text labels on the bars
    for bar in bars:
        width = bar.get_width()
        axes[1].text(width + 0.01, # Position text slightly outside the bar
                     bar.get_y() + bar.get_height()/2., # Center vertically
                     f'{width:.3f}', # Format text
                     va='center')

    # Add a grid for easier reading of probability values
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    # --- Final Touches ---
    plt.tight_layout() # Adjust layout to prevent overlap

    # Save or show the plot
    if output_path:
        try:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"\nPrediction plot saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving plot to {output_path}: {e}")
            print("Displaying plot instead.")
            plt.show()
    else:
        print("\nDisplaying prediction plot...")
        plt.show()

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict soil type from a single image and optionally plot results.")
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument(
        "--model_path",
        default='/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass/best_resnet50_soil_model.pth',
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--plot_output",
        default=None, # Changed default to None (show plot by default)
        # default="prediction_plot.png", # Or set a default filename to always save
        help="Path to save the output plot image (e.g., 'result.png'). If not specified, the plot will be displayed interactively."
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="If set, prevents the plot from being generated or displayed."
    )
    args = parser.parse_args()

    # --- Configuration ---
    MODEL_PATH = args.model_path
    IMAGE_PATH = args.image_path
    PLOT_OUTPUT_PATH = args.plot_output
    NO_PLOT = args.no_plot

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

    # --- Display Text Results ---
    if prediction_result.get('error'):
        print(prediction_result['error'])
        sys.exit(1) # Exit if prediction failed
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

    # --- Generate and Show/Save Plot ---
    if not NO_PLOT:
        plot_prediction_results(IMAGE_PATH, prediction_result, PLOT_OUTPUT_PATH)
    else:
        print("\nPlotting skipped due to --no_plot flag.")