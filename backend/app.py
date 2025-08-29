from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
import uuid

app = Flask(__name__)
CORS(app)

# ---------------------------
# Load Model
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_weights", "pneumonia_model.pth")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# ---------------------------
# Grad-CAM Helpers
# ---------------------------
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# ---------------------------
# Image Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3 channels
])

def generate_gradcam(image_tensor):
    output = model(image_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    fmap = feature_maps[0][0].detach().numpy()
    grad = gradients[0][0].detach().numpy()

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam, pred_class

# ---------------------------
# Flask Route
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    img_tensor.requires_grad_()

    # Clear previous hooks data
    feature_maps.clear()
    gradients.clear()

    # Register hooks on last conv block
    hook_fwd = model.layer4[-1].register_forward_hook(forward_hook)
    hook_bwd = model.layer4[-1].register_backward_hook(backward_hook)

    # Forward pass + softmax
    output = model(img_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, 1)

    # Generate Grad-CAM
    heatmap, _ = generate_gradcam(img_tensor)

    # Remove hooks
    hook_fwd.remove()
    hook_bwd.remove()

    # Prepare heatmap
    img = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save Grad-CAM image
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)

    img_id = f"{uuid.uuid4().hex[:8]}.jpg"
    heatmap_path = os.path.join(static_dir, img_id)
    cv2.imwrite(heatmap_path, superimposed)

    result = {
        'prediction': 'Pneumonia' if predicted_class.item() == 1 else 'Normal',
        'confidence': f"{confidence.item() * 100:.2f}%",
        'heatmap_url': f"http://127.0.0.1:5000/static/{img_id}"
    }

    return jsonify(result)

# ---------------------------
# Run App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
