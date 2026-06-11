# app_ENB3.py

import json
import streamlit as st
import torch
import timm
import numpy as np

from PIL import Image
from torchvision import transforms

from sensors_utils import simulate_sensor_data
from llm_engine import generate_llm_explanation
from gradcam_utils import GradCAM

from dotenv import load_dotenv
import os

load_dotenv()

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿"
)

st.markdown("""
<style>

/* убрать огромный отступ сверху */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
}

/* основной текст */
html, body, [class*="css"] {
    font-size: 24px !important;
}

/* заголовки */
h2 {
    font-size: 42px !important;
}

h3 {
    font-size: 34px !important;
}

/* кнопки */
.stButton > button {
    font-size: 24px !important;
    height: 70px !important;
    width: 100% !important;
    border-radius: 12px;
    font-weight: 600;
}

/* ввод текста */
.stTextInput input {
    font-size: 24px !important;
    height: 60px !important;
}

/* загрузка файла */
section[data-testid="stFileUploader"] {
    font-size: 24px !important;
}

/* все подписи */
label, p, span {
    font-size: 24px !important;
}

/* прогресс */
.stProgress > div > div {
    height: 28px !important;
}

/* json */
pre {
    font-size: 20px !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;margin-bottom:15px;'>

<h1 style='
font-size:48px;
font-weight:700;
color:#2E8B57;
margin-bottom:0px;
'>
🌿 PlantGuard AI - EfficientNet-B3
</h1>

<p style='
font-size:22px;
color:#666;
margin-top:0;
'>
EfficientNet-B3 Disease Detection System
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "EfficientNet-B3\checkpoints\efficientnet_b3_final.pth"
CLASSES_PATH = "classes.json"

IMAGE_SIZE = 300

# =========================================================
# TRANSFORM
# =========================================================

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# LOAD CLASSES
# =========================================================

with open(CLASSES_PATH, "r") as f:

    class_names = json.load(f)

# =========================================================
# FORMAT CLASS
# =========================================================

def format_class(name):

    return (
        name
        .replace("___", " : ")
        .replace("_", " ")
    )

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=len(class_names)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

# =========================================================
# PREDICT
# =========================================================

def predict_top3(image):

    tensor = (
        val_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = model(tensor)

        probs = torch.softmax(outputs, dim=1)

        top_probs, top_idxs = torch.topk(probs, 3)

    results = []

    for prob, idx in zip(top_probs[0], top_idxs[0]):

        results.append({
            "class": class_names[idx.item()],
            "confidence": prob.item() * 100
        })

    return results

# =========================================================
# UPLOAD IMAGE
# =========================================================

uploaded_file = st.file_uploader(
    "Upload plant leaf image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# MAIN
# =========================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

    with st.spinner("Analyzing plant disease..."):

        top3 = predict_top3(image)

        sensors = simulate_sensor_data()

    # =====================================================
    # RESULT
    # =====================================================

    st.subheader(
        f"🌿 Predicted Disease: "
        f"**{format_class(top3[0]['class'])}**"
    )

    st.progress(top3[0]["confidence"] / 100)

    st.write(
        f"Confidence: "
        f"{top3[0]['confidence']:.2f}%"
    )

    # =====================================================
    # TOP 3
    # =====================================================

    st.write("### 🔎 Top-3 Predictions")

    for item in top3:

        st.write(
            f"{format_class(item['class'])}"
            f" — "
            f"{item['confidence']:.2f}%"
        )

    # =====================================================
    # SENSOR DATA
    # =====================================================

    st.write("### 🌡 Sensor Data")

    st.json(sensors)

    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.write("### 🧠 Model Attention (Grad-CAM)")

    input_tensor = (
        val_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    target_layer = model.conv_head

    gradcam = GradCAM(
        model,
        target_layer
    )

    heatmap = gradcam.generate(input_tensor)

    heatmap_img = (
        heatmap * 255
    ).astype(np.uint8)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            heatmap_img,
            caption="White zones show where the model focused",
            width=400
        )

    # =====================================================
    # AI DIAGNOSIS
    # =====================================================

    st.write("### 🤖 AI Diagnosis")

    if st.button("Generate AI Diagnosis"):

        explanation = generate_llm_explanation(
            top3_predictions=top3,
            sensors=sensors
        )

        st.write(explanation)

    # =====================================================
    # USER Q&A
    # =====================================================

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