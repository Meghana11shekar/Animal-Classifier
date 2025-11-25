import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load the trained model
model = tf.keras.models.load_model("animal_classifier_model.keras")

# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Invert the class indices to get index to class name mapping
class_names = {v: k for k, v in class_indices.items()}

# Streamlit page configuration
st.set_page_config(page_title="Animal Classifier", page_icon="🦁", layout="centered", initial_sidebar_state="auto")

# Custom CSS styling for better UI
st.markdown("""
    <style>
    body {
        background-color: #0f1117;
        color: white;
    }
    .main-prediction {
        font-size: 28px;
        font-weight: bold;
        color: #00fa9a;
        background-color: #1c1f26;
        padding: 10px 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .secondary-prediction {
        font-size: 18px;
        color: #95a5a6;
        margin-left: 10px;
    }
    .section-title {
        font-size: 22px;
        color: #ffffff;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p style="font-size:64px; font-weight:900; text-align:center; color:#f39c12;">🐾 Animal Image Classifier</p>', unsafe_allow_html=True)


# Instruction
st.write("Upload an animal image to classify it into one of the 10 classes.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Normalize

    # Show progress while predicting
    with st.spinner('Analyzing image and predicting...'):
        predictions = model.predict(img_array)

    sorted_indices = np.argsort(predictions[0])[::-1]  # Top to bottom

    top_class = class_names[sorted_indices[0]]
    top_conf = predictions[0][sorted_indices[0]] * 100

    # Show progress bar for main prediction confidence
    st.progress(min(int(top_conf), 100))

    # Display main prediction with highlight
    st.markdown(f'<div class="main-prediction">🎯 Prediction: {top_class.capitalize()} ({top_conf:.2f}%)</div>', unsafe_allow_html=True)

    # Display next top 2 predictions
    st.markdown("<div class='section-title'>Other likely predictions:</div>", unsafe_allow_html=True)
    for i in range(1, 3):
        class_name = class_names[sorted_indices[i]]
        conf = predictions[0][sorted_indices[i]] * 100
        st.markdown(f'<div class="secondary-prediction">{i+1}. {class_name.capitalize()} ({conf:.2f}%)</div>', unsafe_allow_html=True)

    st.success("✅ Prediction completed!")




