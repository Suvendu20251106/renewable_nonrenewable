import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# --- Constants ---
CLASS_NAMES = [
    "Non-biodegradable",
    "Biodegradable - Renewable",
    "Biodegradable - Non-renewable"
]

# --- Load model ---
@st.cache_resource
def load_model(model_path="waste_classifier_model.h5"):
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please train and place your model file in the project directory.")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

# --- Preprocess image ---
def preprocess_image(image: Image.Image):
    """Convert image to RGB, resize, normalize, and add batch dimension."""
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Ensure size matches model input
    img_array = np.array(image) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Waste Detection and Classification", layout="centered")
    st.title("♻️ Waste Detection and Classification")
    
    st.write(
        """
        Upload an image of a waste item. The app will classify it as:
        - **Non-biodegradable**  
        - **Biodegradable - Renewable**  
        - **Biodegradable - Non-renewable**  

        🌿 Renewable biodegradable waste earns you recycling points!
        """
    )

    model = load_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Classifying the image..."):
            try:
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                predicted_class_idx = np.argmax(predictions)
                confidence = predictions[0][predicted_class_idx]
                predicted_class = CLASS_NAMES[predicted_class_idx]

                st.markdown(f"### Prediction: *{predicted_class}*")
                st.markdown(f"**Confidence:** {confidence:.2f}")

                # Show all class probabilities
                st.write("#### Probabilities for each class:")
                for i, class_name in enumerate(CLASS_NAMES):
                    st.write(f"{class_name}: {predictions[0][i]:.2f}")

                # Incentive mechanism
                if predicted_class == "Biodegradable - Renewable":
                    st.success("🌿 This is renewable biodegradable waste! You earned 10 recycling points! 🎉")
                else:
                    st.info("Please dispose of this waste properly.")

            except Exception as e:
                st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()
