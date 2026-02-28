import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from rgbd_model import RGBDUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    model = RGBDUNet().to(device)
    model.load_state_dict(torch.load("best_rgbd_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

st.title("🛣️ Pothole Detection + Depth Measurement")

rgb_file = st.file_uploader("Upload RGB Image", type=["jpg", "png", "jpeg"])
depth_file = st.file_uploader("Upload Depth (.npy file)", type=["npy"])

if rgb_file is not None:

    image = Image.open(rgb_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    if depth_file is not None:
        depth_array = np.load(depth_file)
        depth_array = cv2.resize(depth_array, (640, 640))
        depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
        depth_tensor = torch.tensor(depth_array).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        depth_tensor = torch.zeros((1,1,640,640)).to(device)

    # Inference
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor, depth_tensor))

    mask = (pred > 0.5).float()
    mask_np = mask.squeeze().cpu().numpy()

    image_np = np.array(image.resize((640,640)))

    # -------------------------
    # Find Bounding Box
    # -------------------------
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image_np, (x,y), (x+w, y+h), (0,255,0), 2)

        # Area in pixels
        area = cv2.contourArea(cnt)

        # Depth measurements
        if depth_file is not None:
            pothole_pixels = depth_array[mask_np > 0.5]
            avg_depth = np.mean(pothole_pixels)
            max_depth = np.max(pothole_pixels)
        else:
            avg_depth = 0
            max_depth = 0

        st.write("📦 Bounding Box:")
        st.write(f"Width: {w} px")
        st.write(f"Height: {h} px")
        st.write(f"Area: {area:.2f} px²")

        if depth_file is not None:
            st.write(f"Average Depth: {avg_depth:.4f}")
            st.write(f"Max Depth: {max_depth:.4f}")

    st.image(image_np, caption="Detection + Bounding Box", use_container_width=True)