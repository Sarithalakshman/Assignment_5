# STREAMLIT CODING (Model Deployment)
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
#Use the modern Keras utility for array conversion
from tensorflow.keras.utils import img_to_array 

# define constant
IMG_WIDTH, IMG_HEIGHT = 224, 224


MODEL_PATH = r"C:\Users\sarit\OneDrive\Desktop\multiclass_fish\vgg16_classifier_model.h5"


CLASS_NAMES = [
    'animal fish', 
    'animal fish bass', 
    'fish sea_food black_sea_sprat', 
    'fish sea_food gilt_head_bream', 
    'fish sea_food hourse_mackerel', 
    'fish sea_food red_mullet', 
    'fish sea_food red_sea_bream', 
    'fish sea_food sea_bass', 
    'fish sea_food shrimp', 
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]


#  Model Loading Function (Cached for Speed) 
@st.cache_resource
def load_model_cached(model_path):
    """Loads the Keras model and caches it to prevent reloading."""
    try:
        
        model = load_model(model_path, compile=False) 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

#Prediction Function
def preprocess_and_predict(uploaded_file, model):
    """Preprocesses the image and makes a prediction."""
    # 1. Open and resize the image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # 2. Convert to array and add batch dimension
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 3. Rescale to match training preprocessing
    img_array = img_array / 255.0 
    
    # 4. Predict
    predictions = model.predict(img_array)
    
    # 5. Decode predictions
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    return predicted_class, confidence, img

# Streamlit App Interface
st.title("🐟 Multiclass Fish Image Classifier")
# Updated description to reflect the chosen model
st.markdown(f"Upload a fish image to get a real-time prediction from the **VGG16** Transfer Learning model.")

# Load the model
model = load_model_cached(MODEL_PATH)

if model is not None:
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        display_img = Image.open(uploaded_file)
        st.subheader("Uploaded Image:")
        st.image(display_img, caption='Image for Classification', use_column_width=True)

        with st.spinner('Classifying image...'):
            # Run prediction
            predicted_class, confidence, _ = preprocess_and_predict(uploaded_file, model)
        
        # Display results
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
else:
    st.warning(f"Model file '{MODEL_PATH}' not found. Please ensure the best model is trained and saved in the same directory.")