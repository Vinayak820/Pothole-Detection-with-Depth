import torch
import cv2
import numpy as np
from rgbd_model import RGBDUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Model
# -------------------------
model = RGBDUNet().to(device)
model.load_state_dict(torch.load("best_rgbd_model.pth", map_location=device))
model.eval()

# -------------------------
# Video Path
# -------------------------
video_path = "test_video.mp4"  # change this
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)

# Get original resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 20.0, (width, height))

print("Starting video processing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()

    # Resize for model
    frame_resized = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0

    rgb_tensor = torch.tensor(frame_rgb).permute(2,0,1).unsqueeze(0).to(device)

    # Use zero depth if no depth video
    depth_tensor = torch.zeros((1,1,640,640)).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(rgb_tensor, depth_tensor))

    mask = (pred > 0.5).float()
    mask_np = mask.squeeze().cpu().numpy()

    # Resize mask back to original frame
    mask_resized = cv2.resize(mask_np, (width, height))
    mask_uint8 = (mask_resized * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area < 500:  # filter small noise
            continue

        cv2.rectangle(original_frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.putText(original_frame,
                    f"Pothole Area: {int(area)} px",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

    # Show frame
    cv2.imshow("Pothole Detection", original_frame)
    out.write(original_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Saved as output_video.mp4")