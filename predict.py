import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("leaf_model.h5")

# IMPORTANT: must match training order exactly
classes = ['alovera', 'banana', 'mango', 'mint', 'Neem', 'Tulasi']

# Load image
img = image.load_img("test.jpg", target_size=(128, 128))

# Convert image to array
img_array = image.img_to_array(img)

# Normalize (VERY IMPORTANT)
img_array = img_array / 255.0

# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# Predict
result = model.predict(img_array)

# Get predicted class
predicted_class = classes[np.argmax(result)]

# Medicinal leaves (match class names EXACTLY)
medicinal_leaves = ['alovera', 'mint', 'Neem', 'Tulasi']

print("Detected Leaf:", predicted_class)

# Leaf info
leaf_info = {
    "Neem": {
        "uses": "Skin care, blood purification",
        "treatment": "Used for acne and skin infection"
    },
    "Tulasi": {
        "uses": "Cold and cough treatment",
        "treatment": "Used in herbal tea"
    },
    "mint": {
        "uses": "Digestion improvement",
        "treatment": "Used for stomach pain"
    },
    "alovera": {
        "uses": "Burn treatment",
        "treatment": "Applied on burns"
    }
}

# Check medicinal or not
if predicted_class in medicinal_leaves:
    print("Type: Medicinal Leaf 🌿")

    info = leaf_info[predicted_class]
    print("Uses:", info["uses"])
    print("Treatment:", info["treatment"])

else:
    print("Type: Non-Medicinal Leaf ❌")
