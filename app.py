import streamlit as st
from streamlit_drawable_canvas import st_canvas 
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn_model.keras')

model = load_model()

st.title("Draw a Digit and Get Prediction")
st.write("Draw a digit (0-9) in the box below and click 'Predict'.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
)

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
    img = img.resize((28, 28))
    
    st.image(img, caption="Processed Input", width=100)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.success(f"🎯 Predicted Digit: **{digit}**")
        st.info(f"📊 Confidence: **{confidence:.1f}%**")
