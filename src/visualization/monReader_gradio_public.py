import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

# Load pre-trained model
model = load_model('../models/MobileNet.keras')

# Define a function that will take an image, resize it, and return predictions
def classify_image(img):
    img = Image.fromarray(img).resize((64, 64))  # Correctly resize the image to 64x64 pixels
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    probability = float(prediction[0][0])
    class_label = "not flip" if probability > 0.5 else "flip"
    if class_label == "flip":
        probability = 1 - probability
        
    return class_label, probability

# Create the Gradio interface
iface = gr.Interface(
    title='Page Flip Detector',
    fn=classify_image,
    inputs=gr.Image(),  # Use gr.Image() directly
    outputs=[gr.Text(label="Class"), gr.Text(label="Probability")]  # Use gr.Text() directly
)

# Launch the interface
iface.launch(share=True)

