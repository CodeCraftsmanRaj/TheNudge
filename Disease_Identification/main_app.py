# # Library imports
# import numpy as np
# import streamlit as st
# import cv2
# from keras.models import load_model
# import tensorflow as tf

# # Loading the Model
# model = load_model('plant_disease_model.h5')

# # Name of Classes
# CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# # Setting Title of App
# st.title("Plant Disease Detection")
# st.markdown("Upload an image of the plant leaf")

# # Uploading the dog image
# plant_image = st.file_uploader("Choose an image...", type = "jpg")
# submit = st.button('predict Disease')

# # On predict button click
# if submit:
#     if plant_image is not None:
#         # Convert the file to an opencv image.
#         file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
#         opencv_image = cv2.imdecode(file_bytes, 1)

#         # Displaying the image
#         st.image(opencv_image, channels="BGR")
#         st.write(opencv_image.shape)

#         # Resizing the image
#         opencv_image = cv2.resize(opencv_image, (256, 256))

#         # Convert image to 4 Dimension
#         opencv_image.shape = (1, 256, 256, 3)

#         #Make Prediction
#         Y_pred = model.predict(opencv_image)
#         result = CLASS_NAMES[np.argmax(Y_pred)]
#         st.title(str("This is "+result.split('-')[0]+ " leaf with " +  result.split('-')[1]))

#
# # Library imports
# import numpy as np
# import cv2
# from keras.models import load_model
# import tensorflow as tf
# import os # Import os module for path validation
#
# # --- Configuration ---
# MODEL_PATH = 'plant_disease_model.h5'
# CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Early_blight', 'Corn-Common_rust') # Corrected typo 'Barly' to 'Early' based on common datasets
# IMAGE_SIZE = (256, 256)
#
# # --- Load Model ---
# # Wrap in a try-except block for better error handling
# try:
#     model = load_model(MODEL_PATH)
#     print(f"Model '{MODEL_PATH}' loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     # Depending on your application, you might want to exit or handle this differently
#     exit() # Exit if model loading fails
#
# # --- Prediction Function ---
# def predict_plant_disease(image_path):
#     """
#     Predicts the plant disease from an image file.
#
#     Args:
#         image_path (str): The path to the image file.
#
#     Returns:
#         str: A string describing the predicted plant and disease,
#              or an error message if prediction fails.
#     """
#     # 1. Validate image path
#     if not os.path.exists(image_path):
#         return f"Error: Image path not found: {image_path}"
#
#     # 2. Read and preprocess the image
#     try:
#         # Read the image using OpenCV
#         opencv_image = cv2.imread(image_path)
#
#         if opencv_image is None:
#             return f"Error: Could not read image file: {image_path}"
#
#         # Resize the image
#         resized_image = cv2.resize(opencv_image, IMAGE_SIZE)
#
#         # --- IMPORTANT: Add Normalization if your model expects it ---
#         # Most models expect pixel values normalized between 0 and 1
#         # Uncomment the following line if your model was trained with normalized data:
#         # normalized_image = resized_image / 255.0
#         # Use normalized_image below instead of resized_image if you uncomment
#
#         # Convert image to 4 Dimensions (add batch dimension)
#         # Use np.expand_dims for clarity and standard practice
#         input_image = np.expand_dims(resized_image, axis=0) # Shape: (1, 256, 256, 3)
#         # Ensure the dtype matches the model's expected input type (often float32)
#         input_image = np.array(input_image, dtype=np.float32) # Use normalized_image here if applicable
#
#     except Exception as e:
#         return f"Error during image preprocessing: {e}"
#
#     # 3. Make Prediction
#     try:
#         predictions = model.predict(input_image)
#         predicted_index = np.argmax(predictions[0]) # Get index of the highest probability
#
#         if predicted_index < 0 or predicted_index >= len(CLASS_NAMES):
#              return f"Error: Predicted index ({predicted_index}) is out of bounds for CLASS_NAMES."
#
#         predicted_class_name = CLASS_NAMES[predicted_index]
#
#         # 4. Format the result string
#         parts = predicted_class_name.split('-')
#         if len(parts) == 2:
#             result_string = f"Prediction: This is a {parts[0]} leaf with {parts[1]}."
#         else:
#             # Fallback if the class name format is unexpected
#             result_string = f"Prediction: {predicted_class_name}"
#
#         return result_string
#
#     except Exception as e:
#         return f"Error during prediction: {e}"
#
# # --- Example Usage ---
# if __name__ == "__main__":
#     # Replace with the actual path to your test image
#     test_image_path = '/home/raj_99/Projects/PragatiAI/Disease/istockphoto-1991307372-612x612.jpg' # <--- CHANGE THIS
#
#     # Check if the placeholder path needs changing
#     if test_image_path == 'path/to/your/test_image.jpg':
#        print("="*30)
#        print("Please update 'test_image_path' in the code")
#        print("with the actual path to your image file.")
#        print("="*30)
#     else:
#         prediction_result = predict_plant_disease(test_image_path)
#         print(prediction_result)
#
#         # Example with a potentially non-existent file
#         # print("\nTesting with a non-existent file:")
#         # non_existent_path = 'path/to/non_existent_image.jpg'
#         # print(predict_plant_disease(non_existent_path))



# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
from PIL import Image # Use PIL for easier handling with Streamlit

# --- Page Configuration (Set Title, Icon, Layout) ---
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿", # Add a relevant emoji icon
    layout="wide", # Use wide layout for better spacing
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Load Model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_keras_model():
    try:
        model = load_model('plant_disease_model.h5', compile=False) # Add compile=False if you get optimizer errors on load
        # Optionally re-compile if needed for specific metrics later, but often not required for just prediction
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_keras_model()

# Name of Classes (ensure order matches model output)
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Early_blight', 'Corn-Common_rust') # Corrected Potato disease name based on common datasets

# --- Sidebar ---
with st.sidebar:
    st.header("🌿 Plant Disease Detector")
    st.markdown("Upload an image of a plant leaf (Tomato, Potato, or Corn) to identify potential diseases.")
    st.markdown("---")
    plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) # Allow more types
    st.markdown("---")
    st.caption("Model trained on specific dataset.") # Add any relevant info

# --- Main Page Layout ---
st.title("Plant Leaf Disease Analysis")
st.markdown("Upload an image via the sidebar and click 'Analyze Leaf' to check for common diseases.")

col1, col2 = st.columns(2) # Create two columns

if plant_image is not None and model is not None:
    # Display uploaded image preview in the first column
    with col1:
        st.subheader("Uploaded Leaf Image")
        try:
            # Use PIL to open the image directly from the uploader buffer
            image = Image.open(plant_image).convert("RGB") # Ensure RGB format
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
            image = None # Prevent further processing if display fails

    # Analysis button and results in the second column
    with col2:
        st.subheader("Analysis Results")
        analyze_button = st.button('Analyze Leaf')

        if analyze_button and image is not None:
            with st.spinner('Processing the image and predicting...'):
                try:
                    # --- Preprocessing ---
                    # Resize using PIL (maintains aspect ratio better if needed, but here we force size)
                    img_resized = image.resize((256, 256)) # Resize to model's expected input size

                    # Convert PIL Image to NumPy array
                    img_array = np.array(img_resized)

                    # Ensure it's float32 and normalize if the model expects it (common practice)
                    # If your model was trained on images scaled 0-1:
                    # img_array = img_array.astype('float32') / 255.0
                    # If not, just ensure the dtype is suitable for the model (often float32)
                    img_array = img_array.astype('float32')

                    # Add batch dimension -> (1, 256, 256, 3)
                    img_batch = np.expand_dims(img_array, axis=0)

                    # --- Make Prediction ---
                    predictions = model.predict(img_batch)
                    predicted_class_index = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100 # Get confidence score

                    if predicted_class_index < len(CLASS_NAMES):
                         result = CLASS_NAMES[predicted_class_index]
                         plant_name = result.split('-')[0]
                         disease_name = result.split('-')[1].replace('_', ' ') # Replace underscore for display
                         st.success(f"**Prediction:** {plant_name} leaf with **{disease_name}**")
                         st.info(f"**Confidence:** {confidence:.2f}%")

                         # Optional: Add more info based on the disease
                         if disease_name == "Bacterial spot":
                             st.markdown("Bacterial spot often appears as small, water-soaked spots that may turn brown or black.")
                         elif disease_name == "Early blight":
                             st.markdown("Early blight typically shows dark lesions, often with concentric rings (target spots).")
                         elif disease_name == "Common rust":
                             st.markdown("Common rust appears as small, reddish-brown pustules, often on both leaf surfaces.")
                    else:
                         st.error("Error: Predicted class index is out of bounds.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        elif analyze_button and image is None:
             st.warning("Please upload a valid image first.")

elif model is None:
    st.error("Model could not be loaded. Please check the model file and logs.")
else:
    # Message when no image is uploaded yet
    st.info("Please upload an image using the sidebar to begin analysis.")