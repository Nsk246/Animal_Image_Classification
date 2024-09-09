import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Load the saved model
model = keras.models.load_model('animal_classification_model.h5')

# Define the image dimensions
img_width, img_height = 224, 224

# Get the class labels from the train_images generator
class_labels = {
    0: 'Bear',
    1: 'Bird',
    2: 'Cat',
    3: 'Cow',
    4: 'Deer',
    5: 'Dog',
    6: 'Dolphin',
    7: 'Elephant',
    8: 'Giraffe',
    9: 'Horse',
    10: 'Kangaroo',
    11: 'Lion',
    12: 'Panda',
    13: 'Tiger',
    14: 'Zebra'
}

# Define the image preprocessing function
def preprocess_image(img):
    img = Image.fromarray(img)
    img = img.resize((img_width, img_height))
    #img = np.array(img) / 255.0  # Normalize the image
    return img

# Define the function to make predictions
def make_prediction(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    confidence_score = np.max(predictions, axis=1)

    # Return the predicted label and confidence score
    return class_labels[predicted_class[0]], float(confidence_score[0])

# Custom CSS for additional styling
custom_css = """
    body {
        background: linear-gradient(to right, #a1c4fd, #c2e9fb);
        font-family: 'Roboto', sans-serif;
        color: #333;
    }
    .gr-button {
        background-color: #4CAF50;
        color: white;
        border-radius: 30px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .gr-button:hover {
        background-color: #45a049;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    }
    .gr-textbox, .gr-number {
        font-size: 16px;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        background: #f9f9f9;
    }
    .gr-textbox {
        margin-top: 20px;
    }
    .gr-number {
        margin-top: 10px;
    }
    .gr-markdown {
        font-size: 18px;
        color: #555;
    }
    .gr-input-container {
        margin-top: 20px;
    }
    .gr-output-container {
        margin-top: 20px;
    }
"""

# Create a Gradio interface with a custom theme and CSS
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    with gr.Column():
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 28px; font-weight: bold; color: #F6DCAC;">
                <span style="color: #F6DCAC;">Animal Classification</span> App 🐾
            </div>
            <div style="text-align: center; font-size: 18px; margin-bottom: 30px; color: #555;">
                Upload an image to classify it into one of the 15 animal classes.
            </div>
            """,
            elem_id="title"
        )

    img_input = gr.Image(label="Upload an image", type="numpy", elem_id="img_input")

    predict_button = gr.Button("Predict", elem_id="predict_button")

    output_label = gr.Textbox(label="Predicted class label", elem_id="output_label")
    output_confidence = gr.Number(label="Confidence score", elem_id="output_confidence")

    predict_button.click(
        make_prediction,
        inputs=img_input,
        outputs=[output_label, output_confidence]
    )

    # Inject custom CSS
    gr.HTML(f"""
        <style>
            {custom_css}
        </style>
    """)

demo.launch()
