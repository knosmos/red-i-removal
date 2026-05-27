import sys
import torch
import numpy as np
import cv2

from model import RedIRemover

assert len(sys.argv) == 2, "Usage: python run_model.py <image_path>"

# Load the model
model = RedIRemover()
model.load_state_dict(torch.load("outputs/best_model.pth"))
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load image
img = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

# Run
with torch.no_grad():
    img = img.to("cuda" if torch.cuda.is_available() else "cpu")
    output = model(img)
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    output = (output * 255).astype(np.uint8)

# Show and save
cv2.imwrite("outputs/inference_output.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
cv2.imshow("Output", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
