import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image

# Define and register the custom metric
@tf.keras.utils.register_keras_serializable()
def mae(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)

# Load the pre-trained model with custom objects
custom_objects = {
    'mae': mae,
}
model = load_model('model_pretrain.h5', custom_objects=custom_objects)

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert the image to RGB if it's a different format
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the image to the required size (128x128 in this case)
    image = image.resize((128, 128))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the image
    image_array = image_array / 255.0
    # Expand dimensions to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define a function to predict age and gender
def predict_age_gender(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    age = predictions[0][0]
    return age
    #gender = 'Male' if predictions[1][0] > 0.5 else 'Female'
    #return age, gender

# Streamlit UI
st.set_page_config(
    page_title="Age and Gender Prediction",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧑 Age and Gender Prediction")
st.write("Upload an image, and the model will predict the age (and gender) of the person in the image.")

st.sidebar.header("Image Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True, channels="RGB")
    st.write("")

    if st.button('Classify'):
        with st.spinner('Classifying...'):
            age = predict_age_gender(image)
            st.success(f"Predicted Age: {age:.2f} years")
            # st.write(f"Predicted Gender: {gender}")
else:
    st.info("Please upload an image to start.")

# Add some spacing at the bottom
st.write("\n" * 10)

# Add footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #f1f1f1;
        font-size: 14px;
        color: #333;
    }
    </style>
   
    """,
    unsafe_allow_html=True
)
