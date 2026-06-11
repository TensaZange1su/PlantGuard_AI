# vision_utils.py

import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224


# ---------- Load classes ----------
def load_classes(path="classes.json"):

    with open(path, "r") as f:
        classes = json.load(f)

    return classes


CLASSES = load_classes()


# ---------- Image preprocessing ----------
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------- Load model ----------
def load_model(model_class, model_path):

    model = model_class(num_classes=len(CLASSES)).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(state_dict, strict=False)

    model.eval()

    return model


# ---------- Prediction ----------
def predict_top3(model, image: Image.Image):

    image = image.convert("RGB")

    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)

    top_probs, top_idx = torch.topk(probs, 3)

    results = []

    for i in range(3):

        class_id = top_idx[0][i].item()

        results.append({
            "class": CLASSES[class_id],
            "confidence": float(top_probs[0][i]) * 100
        })

    return results