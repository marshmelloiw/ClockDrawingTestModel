import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("VGG16.h5")

def predict_drawing(image):
    image = image.convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_labels = ['0_no_clock', '1_severe_vis', '2_mod_vis_xhands',
                    '3_hands_vis_errors', '4_minor_VIS_errors', '5_perfect_clock']
    return {label: float(prediction[0][i]) for i, label in enumerate(class_labels)}

demo = gr.Interface(
    fn=predict_drawing,
    inputs=gr.Sketchpad(shape=(224, 224)),
    outputs=gr.Label(num_top_classes=3),
    title="Clock Drawing Classifier",
    description="Lütfen saat çizimi yapın. Model tahmin etsin."
)

demo.launch()
