"""
Task 1.2 - verify YOLOv5 on the EC2 instance using the SAME loading path the
Lambda function uses (local torch.hub load, pretrained yolov5s weights).

    python ec2_verify_yolov5.py --yolo ~/yolov5 --image ~/yolov5/data/images/bus.jpg
"""
import argparse, time
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--yolo", default="yolov5")
ap.add_argument("--weights", default=None)
ap.add_argument("--image", required=True)
ap.add_argument("--out", default="verify_output.jpg")
a = ap.parse_args()
weights = a.weights or f"{a.yolo}/yolov5s.pt"

t = time.time()
model = torch.hub.load(a.yolo, "custom", path=weights, source="local")
print(f"model loaded in {time.time() - t:.2f}s - {len(model.names)} COCO classes, torch {torch.__version__}")

img = np.array(Image.open(a.image).convert("RGB"))
t = time.time()
res = model(img, size=640)
print(f"inference: {(time.time() - t) * 1000:.0f} ms")
print(res.pandas().xyxy[0][["name", "confidence", "xmin", "ymin", "xmax", "ymax"]].round(2).to_string())
Image.fromarray(res.render()[0]).save(a.out)
print(f"annotated image written to {a.out}")
