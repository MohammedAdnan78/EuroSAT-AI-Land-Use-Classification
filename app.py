import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import gradio as gr

IMAGE_SIZE = 64
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.35):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(num_classes=len(class_names)).to(device)

weights_path = "eurosat_cnn_best.pt"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    print("Warning: eurosat_cnn_best.pt not found. Train the notebook first and copy the saved weights here.")

model.eval()

FALLBACK_LAND_USE_INFO = {
    "AnnualCrop": "Annual crop areas are agricultural lands planted and harvested within one growing season. They are important for food production and seasonal farming analysis.",
    "Forest": "Forest regions support biodiversity, carbon storage, water regulation, and climate stability. Satellite monitoring helps detect deforestation and recovery.",
    "HerbaceousVegetation": "Herbaceous vegetation includes grass-like or low-growing plant cover. It can indicate natural vegetation, grazing areas, or transitional land.",
    "Highway": "Highways are transportation infrastructure. Detecting them helps with urban planning, logistics, and land-use change analysis.",
    "Industrial": "Industrial areas often include factories, warehouses, and large built structures. They indicate economic activity and urban development.",
    "Pasture": "Pasture land is commonly used for grazing animals. It is important for livestock agriculture and rural land management.",
    "PermanentCrop": "Permanent crops include orchards, vineyards, and plantations. These areas often show regular planting patterns in satellite images.",
    "Residential": "Residential areas contain housing and local infrastructure. Detecting them helps track urban expansion and population-related land use.",
    "River": "Rivers are freshwater systems. Monitoring rivers supports flood-risk analysis, erosion tracking, and water-resource management.",
    "SeaLake": "Sea and lake regions represent large water bodies. Monitoring them supports environmental protection, water planning, and climate analysis."
}

def get_ai_land_use_explanation(predicted_class, confidence, top3_text):
    fallback = FALLBACK_LAND_USE_INFO.get(
        predicted_class,
        "This land-use class can be analyzed through satellite imagery for planning and monitoring."
    )

    if not os.getenv("OPENAI_API_KEY"):
        return "OpenAI API key not found. Fallback explanation:\n\n" + fallback

    try:
        from openai import OpenAI
        client = OpenAI()
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        prompt = f"""
A PyTorch EuroSAT satellite image classifier predicted:
Class: {predicted_class}
Confidence: {confidence:.2%}
Top predictions: {top3_text}

Give concise additional information:
1. What this land-use class represents.
2. Why it matters in satellite monitoring.
3. One real-world application.
Keep it under 120 words.
"""

        response = client.responses.create(
            model=model_name,
            input=prompt
        )
        return response.output_text

    except Exception as e:
        return f"OpenAI call failed. Fallback explanation shown.\nError: {e}\n\n{fallback}"

@torch.no_grad()
def classify_satellite_image(input_image):
    if input_image is None:
        return "No image uploaded.", "", ""

    pil_image = input_image.convert("RGB")
    image_tensor = eval_transform(pil_image).unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, k=3)

    predicted_index = top_indices[0].item()
    predicted_class = class_names[predicted_index]
    confidence = top_probs[0].item()

    top3 = [
        f"{class_names[idx.item()]}: {prob.item():.2%}"
        for prob, idx in zip(top_probs, top_indices)
    ]
    top3_text = ", ".join(top3)

    ai_info = get_ai_land_use_explanation(predicted_class, confidence, top3_text)

    prediction_text = (
        f"Model: CNN\n"
        f"Predicted Class: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )

    return prediction_text, top3_text, ai_info

demo = gr.Interface(
    fn=classify_satellite_image,
    inputs=gr.Image(type="pil", label="Upload Satellite Image"),
    outputs=[
        gr.Textbox(label="Model Prediction"),
        gr.Textbox(label="Top 3 Predictions"),
        gr.Textbox(label="AI Additional Information")
    ],
    title="AI-Enhanced EuroSAT Satellite Land-Use Classifier",
    description=(
        "Upload a satellite image. The PyTorch CNN predicts the land-use class, "
        "then an OpenAI model explains the predicted land-use category."
    )
)

if __name__ == "__main__":
    demo.launch()
