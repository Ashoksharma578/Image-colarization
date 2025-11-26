import torch
import cv2
import numpy as np
from skimage.color import lab2rgb, rgb2lab
from model import UNet

# --------------------------
# Load Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = UNet().to(device)
model.load_state_dict(torch.load("colorizer_model.pth", map_location=device))
model.eval()

# --------------------------
# Colorize Function
# --------------------------
def colorize_image(img_path, save_path="output.png"):
    # Load image
    image = cv2.imread(img_path)

    if image is None:
        print("[ERROR] Could not load:", img_path)
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))

    # Convert to LAB
    lab = rgb2lab(image).astype("float32")
    L = lab[:, :, 0] / 100.0
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device)

    # Predict ab channels
    with torch.no_grad():
        ab_pred = model(L_tensor)[0].cpu().numpy()
    ab_pred = ab_pred.transpose((1, 2, 0)) * 128

    # Combine L + predicted ab
    L = L * 100
    lab_out = np.zeros((256, 256, 3))
    lab_out[:, :, 0] = L
    lab_out[:, :, 1:] = ab_pred

    # Convert back to RGB
    rgb_out = (lab2rgb(lab_out) * 255).astype("uint8")

    # Save
    cv2.imwrite(save_path, cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved colorized image → {save_path}")

# --------------------------
# Run Test
# --------------------------
# Change this to your test image
test_image = "000000000030.jpg"  # <--- put your grayscale image here
colorize_image(test_image, "result_colorized.png")
