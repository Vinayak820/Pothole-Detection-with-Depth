import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from rgb_model import RGBUNet
from rgbd_model import RGBDUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models():
    rgb_model = RGBUNet().to(device)
    rgb_model.load_state_dict(torch.load("best_rgb_model.pth", map_location=device))
    rgb_model.eval()

    rgbd_model = RGBDUNet().to(device)
    rgbd_model.load_state_dict(torch.load("best_rgbd_model.pth", map_location=device))
    rgbd_model.eval()

    return rgb_model, rgbd_model

rgb_model, rgbd_model = load_models()

# -------------------------
# Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# -------------------------
# UI
# -------------------------
st.title("🛣️ Pothole Detection System")
st.write("Upload a road image to detect potholes.")

model_choice = st.selectbox("Choose Model", ["RGB Model", "RGB-D Model"])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    if model_choice == "RGB Model":
        with torch.no_grad():
            pred = torch.sigmoid(rgb_model(img_tensor))
    else:
        # If user doesn't provide depth, use zero depth map
        depth = torch.zeros((1,1,640,640)).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(rgbd_model(img_tensor, depth))

    mask = (pred > 0.5).float()

    # Convert to overlay
    mask_np = mask.squeeze().cpu().numpy()
    image_np = np.array(image.resize((640,640)))

    colored_mask = np.zeros_like(image_np)
    colored_mask[mask_np > 0.5] = [255, 0, 0]

    blended = cv2.addWeighted(image_np, 0.7, colored_mask, 0.3, 0)

    st.image(blended, caption="Detected Pothole", use_container_width=True)