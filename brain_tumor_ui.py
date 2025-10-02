import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

model = load_model("brain_tumor.keras")  # your trained model
# print(model.input_shape)

# Define image size
IMG_SIZE = (150, 150)  

# Streamlit UI
#Page Configuration
st.set_page_config(
    page_title="Brain Tumor Dectection", 
    page_icon="🧠", 
    layout="centered"
    )

# Set image size same as training
IMG_SIZE = (150, 150)

# Define your 4 class labels (change names if needed)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to detect the type of brain tumor")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB") 
    st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

    # Preprocess image
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)  
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    # Prediction
    prediction = model.predict(img_array)  
    class_index = np.argmax(prediction)    
    class_name = CLASS_NAMES[class_index]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"🧠 Predicted Tumor Type: **{class_name}**")
    st.info(f"Confidence: {confidence:.2f}%")

    # Plot probability distribution
    st.write("### Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, prediction[0])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)
