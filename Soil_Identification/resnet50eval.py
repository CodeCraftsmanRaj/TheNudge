import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        label_vector = row[self.label_cols].values.astype('float32')
        label = torch.tensor(label_vector).argmax().item()

        return image, label, row['filename']  # Also return filename for better reporting

# ---------------------- Model Definition ----------------------
class SoilClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=11, pretrained=False):
        super(SoilClassifier, self).__init__()
        
        # Load pre-trained model
        if model_name == 'resnet50':
            base_model = models.resnet50(weights=None)  # No need for pretrained weights here
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
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------- Evaluation Functions ----------------------
def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []
    all_probs = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_filenames.extend(filenames)
            all_probs.extend(probs.cpu().numpy())
    
    eval_time = time.time() - start_time
    
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
    num_classes = len(class_names)
    conf_matrix = confusion_matrix(
        all_labels, all_preds, 
        labels=list(range(num_classes))
    )
    
    print(f"🧪 Test Accuracy: {test_acc:.4f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print("\nClassification Report:")
    print(class_report)
    
    # Print missing classes if any
    all_class_indices = set(range(len(class_names)))
    missing_indices = all_class_indices - set(present_class_indices)
    if missing_indices:
        missing_classes = [class_names[i] for i in missing_indices]
        print(f"\nNote: The following classes were missing in the test predictions or labels: {missing_classes}")
    
    # Create results dictionary for detailed analysis
    results = {
        'filename': all_filenames,
        'true_label': [class_names[label] for label in all_labels],
        'predicted_label': [class_names[pred] for pred in all_preds],
        'correct': [pred == label for pred, label in zip(all_preds, all_labels)]
    }
    
    # Add probability for each class
    for i, class_name in enumerate(class_names):
        results[f'prob_{class_name}'] = [probs[i] for probs in all_probs]
        
    return test_acc, class_report, conf_matrix, results

# ---------------------- Visualization Functions ----------------------
def plot_confusion_matrix(conf_matrix, class_names, output_path="confusion_matrix.png"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_class_distribution(test_loader, class_names, output_path="class_distribution.png"):
    labels_count = {class_name: 0 for class_name in class_names}
    
    # Count occurrences of each class
    for _, labels, _ in test_loader:
        for label in labels.numpy():
            labels_count[class_names[label]] += 1
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(labels_count.keys(), labels_count.values())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Soil Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Test Set')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Class distribution plot saved to {output_path}")

def analyze_misclassifications(results_df, class_names, output_dir="misclassification_analysis"):
    # Create directory for misclassification analysis
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter incorrect predictions
    incorrect_df = results_df[~results_df['correct']]
    
    # Get counts of each type of misclassification
    misclass_counts = incorrect_df.groupby(['true_label', 'predicted_label']).size().reset_index()
    misclass_counts.columns = ['True Class', 'Predicted Class', 'Count']
    misclass_counts = misclass_counts.sort_values('Count', ascending=False)
    
    # Save to CSV
    misclass_counts.to_csv(f"{output_dir}/misclassification_counts.csv", index=False)
    
    # Find the most common misclassifications
    top_misclassifications = misclass_counts.head(10)
    
    # Plot top misclassifications
    plt.figure(figsize=(12, 8))
    plt.bar(
        [f"{row['True Class']} → {row['Predicted Class']}" for _, row in top_misclassifications.iterrows()],
        top_misclassifications['Count']
    )
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Misclassification Type')
    plt.ylabel('Count')
    plt.title('Top 10 Misclassifications')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_misclassifications.png")
    plt.close()
    
    print(f"Misclassification analysis saved to {output_dir}")
    return misclass_counts

# ---------------------- Main Evaluation Function ----------------------
def main():
    # Paths
    base_path = '/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass'
    test_path = os.path.join(base_path, 'test')
    model_path = '/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass/best_resnet50_soil_model.pth'
    output_dir = os.path.join(base_path, 'evaluation_results')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the saved model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    class_names = checkpoint['class_names']
    
    # Initialize model
    model = SoilClassifier(model_name='resnet50', num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully. Validation accuracy at save time: {checkpoint['val_acc']:.4f}")
    
    # Define transform for test images
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset and dataloader
    test_dataset = SoilDataset(os.path.join(test_path, '_classes.csv'), test_path, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Print class distribution in test set
    class_counts = {col: test_dataset.data[col].sum() for col in test_dataset.label_cols}
    print("\nClass distribution in test set:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Evaluate model
    print("\nEvaluating model...")
    test_acc, class_report, conf_matrix, results = evaluate_model(model, test_loader, device, class_names)
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    
    # Generate visualizations
    plot_confusion_matrix(conf_matrix, class_names, output_path=os.path.join(output_dir, 'confusion_matrix.png'))
    plot_class_distribution(test_loader, class_names, output_path=os.path.join(output_dir, 'class_distribution.png'))
    
    # Analyze misclassifications
    misclass_analysis = analyze_misclassifications(
        results_df, 
        class_names, 
        output_dir=os.path.join(output_dir, 'misclassification_analysis')
    )
    
    # Print overall summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: ResNet50")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Detailed results saved to: {output_dir}")
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:  # Check if class exists in test set
            class_correct = np.sum((np.array(all_preds) == i) & class_mask)
            class_accuracy[class_name] = class_correct / np.sum(class_mask)
    
    print("\nAccuracy per class:")
    for class_name, acc in class_accuracy.items():
        print(f"  {class_name}: {acc:.4f}")
    
    return test_acc, results_df

# ---------------------- Sample Prediction Function ----------------------
def predict_single_image(model, image_path, class_names, device):
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted_class = torch.max(output, 1)
        prediction = predicted_class.item()
    
    # Get class probabilities
    probs_dict = {class_names[i]: round(float(prob) * 100, 2) for i, prob in enumerate(probabilities)}
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'predicted_class': class_names[prediction],
        'confidence': float(probabilities[prediction]),
        'all_probabilities': sorted_probs
    }

# ---------------------- Inference Examples ----------------------
def run_sample_inference(model, test_loader, class_names, device, num_samples=5):
    """Run inference on a few random samples from the test set"""
    # Get some random samples
    dataiter = iter(test_loader)
    images, labels, filenames = next(dataiter)
    
    # Select a subset of samples
    selected_indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    print(f"\nRunning inference on {len(selected_indices)} sample images:")
    
    for idx in selected_indices:
        image = images[idx].unsqueeze(0).to(device)
        label = labels[idx].item()
        filename = filenames[idx]
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(image)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probs, 3)
        
        print(f"\nSample: {filename}")
        print(f"True class: {class_names[label]}")
        print(f"Predicted class: {class_names[prediction]}")
        print(f"Prediction {'correct' if prediction == label else 'incorrect'}")
        print("Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())):
            print(f"  {i+1}. {class_names[idx]}: {prob:.4f}")

# ---------------------- Run Evaluation ----------------------
if __name__ == "__main__":
    # Variables for global scope in case of errors
    all_labels = []
    all_preds = []
    
    try:
        test_acc, results_df = main()
        
        # Get example file paths for future inference
        base_path = '/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass'
        test_path = os.path.join(base_path, 'test')
        print("\nFor future reference, you can run inference on individual images like this:")
        print("```python")
        print("from soil_model_evaluation import predict_single_image")
        print("model_path = '/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass/best_resnet50_soil_model.pth'")
        print("image_path = '/path/to/your/soil/image.jpg'  # Replace with actual image path")
        print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        print("checkpoint = torch.load(model_path)")
        print("class_names = checkpoint['class_names']")
        print("model = SoilClassifier(num_classes=len(class_names)).to(device)")
        print("model.load_state_dict(checkpoint['model_state_dict'])")
        print("result = predict_single_image(model, image_path, class_names, device)")
        print("print(f\"Predicted class: {result['predicted_class']} with {result['confidence']:.2%} confidence\")")
        print("```")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    # Variables for global scope in case of errors
    all_labels = []
    all_preds = []
    
    try:
        test_acc, results_df = main()
        
        # Get example file paths for future inference
        base_path = '/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass'
        test_path = os.path.join(base_path, 'test')
        print("\nFor future reference, you can run inference on individual images like this:")
        print("```python")
        print("from soil_model_evaluation import predict_single_image")
        print("model_path = '/home/raj_99/Projects/PragatiAI/Soil/Soil Classification.v4i.multiclass/best_resnet50_soil_model.pth'")
        print("image_path = '/path/to/your/soil/image.jpg'  # Replace with actual image path")
        print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        print("checkpoint = torch.load(model_path)")
        print("class_names = checkpoint['class_names']")
        print("model = SoilClassifier(num_classes=len(class_names)).to(device)")
        print("model.load_state_dict(checkpoint['model_state_dict'])")
        print("result = predict_single_image(model, image_path, class_names, device)")
        print("print(f\"Predicted class: {result['predicted_class']} with {result['confidence']:.2%} confidence\")")
        print("```")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()