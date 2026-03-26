from tensorflow.keras.models import load_model
import  streamlit as st
import os

def save_model(model):

    os.makedirs("models", exist_ok=True)
    model.save("models/brain_tumor_predictor.keras")

@st.cache_resource
def load_keras_model(path):
    """Loads the pre-trained Keras model."""
    try:
        with st.spinner(f"Loading Keras model from {path}..."):
            model = load_model(path)
            st.success("Model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"Error loading model: Could not find or load model at '{path}'. Details: {e}")
        st.stop()
