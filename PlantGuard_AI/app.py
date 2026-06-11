# app.py

import streamlit as st
from PIL import Image
import torch
import numpy as np

from model_arch import EfficientNetB3Model
from vision_utils import load_model, predict_top3, val_transform
from sensors_utils import simulate_sensor_data
from llm_engine import generate_llm_explanation
from gradcam_utils import GradCAM

def format_class(name):
    return name.replace("___", " : ").replace("_", " ")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Plant Disease AI Assistant",
    page_icon="🌿"
)

st.title("🌿 Plant Disease AI Assistant")


# -------- Load Model --------

@st.cache_resource
def get_model():
    model = load_model(EfficientNetB3Model, "EfficientNet-B3/efficientnet_b3.pth")
    return model


model = get_model()


# -------- Upload Image --------

uploaded_file = st.file_uploader(
    "Upload plant leaf image",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)


    # -------- Prediction --------

    with st.spinner("Analyzing with Vision Model..."):

        top3 = predict_top3(model, image)

        sensors = simulate_sensor_data()


    # -------- Prediction Display --------

    st.subheader(
        f"🌿 Predicted Disease: **{format_class(top3[0]['class'])}**"
    )

    st.progress(top3[0]['confidence'] / 100)

    st.write(f"Confidence: {top3[0]['confidence']:.2f}%")

    st.write("### 🔎 Top-3 Predictions")

    for item in top3:
        st.write(f"{item['class']} — {item['confidence']:.2f}%")


    # -------- Sensor Data --------

    st.write("### 🌡 Sensor Data")

    st.json(sensors)


    # -------- Grad-CAM Explainability --------

    st.write("### 🧠 Model Attention (Grad-CAM)")

    input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, model.model.features[-1])

    heatmap = gradcam.generate(input_tensor)

    heatmap_img = (heatmap * 255).astype(np.uint8)

    st.image(
        heatmap_img,
        caption="White zones show where the model focused",
        use_container_width=True
    )


    # -------- AI Diagnosis --------

    st.write("### 🤖 AI Diagnosis")

    if st.button("Generate AI Diagnosis"):

        explanation = generate_llm_explanation(
            top3_predictions=top3,
            sensors=sensors
        )

        st.write(explanation)


    # -------- User Q&A --------

    st.write("### 💬 Ask AI a Question")

    user_question = st.text_input(
        "Ask about this plant condition..."
    )

    if st.button("Ask AI"):

        if user_question.strip():

            answer = generate_llm_explanation(
                top3_predictions=top3,
                sensors=sensors,
                user_question=user_question
            )

            st.write(answer)

        else:

            st.warning("Please enter a question.")