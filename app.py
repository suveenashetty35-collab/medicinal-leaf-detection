import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("leaf_model.h5")

# Class names (keep same order as training)
classes = ['alovera', 'banana', 'mango', 'mint', 'Neem', 'Tulasi']

# Medicinal list (make case consistent)
medicinal_leaves = ['alovera', 'mint', 'Neem', 'Tulasi']

# Title
st.title("🌿 Medicinal Leaf Detection System")

# Upload image
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=250)

    # Resize image
    img = img.resize((128, 128))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize (IMPORTANT)
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    result = model.predict(img_array)

    predicted_class = classes[np.argmax(result)]

    st.write("### Detected Leaf:", predicted_class)

    # Check medicinal or not
    if predicted_class in medicinal_leaves:
        st.success("Medicinal Leaf 🌿")
    else:
        st.error("Non-Medicinal Leaf ❌")