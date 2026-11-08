"""Nakhlah: run the final EdgeNeXt-Base leaf classifier on the Jetson.

Usage (inside the folder that holds the model files):
    python3 nakhlah_classify.py jetson_test_crops          # classify every image in a folder
    python3 nakhlah_classify.py some_leaf.jpg               # classify one image
    python3 nakhlah_classify.py jetson_test_crops --fp16    # faster half-precision on the GPU

Needs in the same folder: edgenext_nakhlah.pt and nakhlah_model.json (from the Colab notebook).
Inputs are leaf crops (one leaf region per image), like the ones the model was trained on.
If the folder has labels.csv (file,true_class), the script also reports accuracy.
"""
import csv, json, os, sys, time

import numpy as np
import timm
import torch
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
info = json.load(open(os.path.join(HERE, "nakhlah_model.json")))
NAMES, SIZE = info["class_names"], info["image_size"]
MEAN = np.array(info["mean"], dtype=np.float32)
STD = np.array(info["std"], dtype=np.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
fp16 = "--fp16" in sys.argv and device == "cuda"

# build the same architecture (no download needed) and load the trained weights
model = timm.create_model(info["timm_name"].split(".")[0], pretrained=False, num_classes=info["num_classes"])
model.load_state_dict(torch.load(os.path.join(HERE, "edgenext_nakhlah.pt"), map_location="cpu"))
model = model.to(device).eval()
print(f"Model loaded on {device.upper()}{' (FP16)' if fp16 else ''} | classes: {NAMES}")


def preprocess(path):
    """Exactly the training preprocessing: RGB, 224x224, ImageNet colours."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((SIZE, SIZE))
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def classify(path):
    x = preprocess(path).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
        p = torch.softmax(model(x).float(), 1)[0].cpu().numpy()
    k = int(p.argmax())
    return NAMES[k], float(p[k])


target = next((a for a in sys.argv[1:] if not a.startswith("--")), "jetson_test_crops")
files = ([os.path.join(target, f) for f in sorted(os.listdir(target)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
         if os.path.isdir(target) else [target])
truth = {}
lab = os.path.join(target, "labels.csv") if os.path.isdir(target) else ""
if lab and os.path.exists(lab):
    truth = {r["file"]: r["true_class"] for r in csv.DictReader(open(lab))}

classify(files[0])                      # warm-up (the first run is always slow)
if device == "cuda":
    torch.cuda.synchronize()

right, times = 0, []
for f in files:
    t0 = time.time()
    cls, conf = classify(f)
    if device == "cuda":
        torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
    name = os.path.basename(f)
    real = truth.get(name)
    mark = "" if real is None else ("  ✓" if real == cls else f"  ✗ (real: {real})")
    right += real == cls
    print(f"{name:32s} -> {cls:20s} {conf:5.1%}{mark}")

print(f"\n{len(files)} images | average {np.mean(times):.1f} ms per image (including loading and preprocessing)")
if truth:
    print(f"Correct: {right} / {len(files)} = {right / len(files):.1%}")
