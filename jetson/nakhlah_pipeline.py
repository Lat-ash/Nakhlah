"""Nakhlah full pipeline on the Jetson:  whole photo -> YOLO26s finds leaves/stems -> each box -> EdgeNeXt disease.

Usage (inside the folder that holds all the model files):
    python3 nakhlah_pipeline.py jetson_test_photos      # every photo in a folder
    python3 nakhlah_pipeline.py some_photo.jpg          # one photo
    python3 nakhlah_pipeline.py jetson_test_photos --yolo nakhlah_yolo26s_leafstem.pt --cfg nakhlah_yolo_leafstem.json

Needs in the same folder:
    a YOLO model (.pt) + its settings (.json)         default: nakhlah_yolo26s.pt + nakhlah_yolo.json
    edgenext_nakhlah.pt + nakhlah_model.json          (final EdgeNeXt, unchanged)

PALM ANSWER ("disease first", if the .json says decision = disease_first):
    every disease found on >= min_leaves leaves with >= min_prob confidence is LISTED with its confidence;
    the palm is HEALTHY only if no disease is found. (Older .json files: majority vote of the leaves.)
Every box is drawn with its disease and confidence; pictures + results.csv go to results/.
If the folder has labels.csv (file,true_class), it also reports how many photos were right.
"""
import collections, csv, json, os, sys, time

import numpy as np
import timm
import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
arg = lambda flag, default: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- tier 1: YOLO26s (leaf / leaflet + stem) ----------
Y = json.load(open(os.path.join(HERE, arg("--cfg", "nakhlah_yolo.json"))))
IMGSZ, CONF = Y["imgsz"], Y["conf"]
PAD, MIN_PX = Y["crop_padding"], Y["min_box_px"]
DISEASE_FIRST = Y.get("decision") == "disease_first"
MIN_PROB, MIN_LEAVES = Y.get("min_prob", 0.8), Y.get("min_leaves", 1)
det = YOLO(os.path.join(HERE, arg("--yolo", "nakhlah_yolo26s.pt")))
PARTS = det.names                                          # e.g. {0: 'leaflet', 1: 'stem'} or {0: 'leaf'}

# ---------- tier 2: EdgeNeXt disease classifier (unchanged) ----------
M = json.load(open(os.path.join(HERE, "nakhlah_model.json")))
NAMES, SIZE = M["class_names"], M["image_size"]
HEALTHY = [n.lower() for n in NAMES].index("healthy")
MEAN, STD = np.array(M["mean"], np.float32), np.array(M["std"], np.float32)
SQUARE = M.get("crop_mode") == "square"            # new EdgeNeXt (trained on YOLO crops) uses square crops
clf = timm.create_model(M["timm_name"].split(".")[0], pretrained=False, num_classes=M["num_classes"])
clf.load_state_dict(torch.load(os.path.join(HERE, "edgenext_nakhlah.pt"), map_location="cpu"))
clf = clf.to(dev).eval()
COLOR = {"fungalSpotting": (255, 140, 0), "fungalWiltBlight": (220, 20, 60), "healthy": (34, 180, 34), "nutrientDeficiency": (255, 215, 0)}
print(f"Models loaded on {dev.upper()} | YOLO parts {list(PARTS.values())}, {IMGSZ} px, conf {CONF} | "
      + ("square crops | " if SQUARE else "") + (f"palm answer: DISEASE FIRST (leaf >= {MIN_PROB:.0%}, >= {MIN_LEAVES} leaf)" if DISEASE_FIRST else "palm answer: majority vote"))


def find_boxes(im):
    """YOLO boxes (most confident first) >= CONF and >= MIN_PX; if none, the single best box."""
    r = det.predict(np.asarray(im)[:, :, ::-1], imgsz=IMGSZ, conf=0.01, max_det=20, verbose=False)[0]
    boxes = sorted(zip(r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().int().tolist(), r.boxes.xyxy.cpu().tolist()), key=lambda b: -b[0])
    keep = [b for b in boxes if b[0] >= CONF and b[2][2] - b[2][0] >= MIN_PX and b[2][3] - b[2][1] >= MIN_PX]
    return keep or boxes[:1]


@torch.no_grad()
def classify(im, boxes):
    """Same crop as training: 10% padding, 224 px, ImageNet colours. Returns EdgeNeXt probabilities per box."""
    if not boxes:
        return np.zeros((0, len(NAMES)))
    W, H = im.size
    xs = []
    for _, _, (x0, y0, x1, y1) in boxes:
        w, h = x1 - x0, y1 - y0
        if SQUARE:          # square around the box centre (no squashing), outside the photo = grey (as in training)
            S = max(1, int(round(max(w, h) * (1 + 2 * PAD))))
            L, T = int(round((x0 + x1) / 2 - S / 2)), int(round((y0 + y1) / 2 - S / 2))
            c = Image.new("RGB", (S, S), (124, 116, 104))
            a, b, r, d = max(0, L), max(0, T), min(W, L + S), min(H, T + S)
            if r > a and d > b:
                c.paste(im.crop((a, b, r, d)), (a - L, b - T))
        else:               # box + 10% padding, stretched to 224 (the original final EdgeNeXt)
            c = im.crop((int(max(0, x0 - PAD * w)), int(max(0, y0 - PAD * h)), int(min(W, x1 + PAD * w)), int(min(H, y1 + PAD * h))))
        xs.append((np.asarray(c.resize((SIZE, SIZE)), np.float32) / 255.0 - MEAN) / STD)
    x = torch.from_numpy(np.stack(xs).transpose(0, 3, 1, 2)).to(dev)
    return torch.softmax(clf(x).float(), 1).cpu().numpy()


def palm_answer(P):
    """-> (main disease name, {disease: confidence}). Healthy only if no disease is found."""
    if len(P) == 0:
        return "no leaf found", {}
    if not DISEASE_FIRST:
        k = collections.Counter(P.argmax(1).tolist()).most_common(1)[0][0]
        return NAMES[k], ({} if k == HEALTHY else {NAMES[k]: float(P[:, k].max())})
    hits = collections.defaultdict(list)
    for p in P:
        k = int(p.argmax())
        if k != HEALTHY and p[k] >= MIN_PROB:
            hits[k].append(float(p[k]))
    found = {k: v for k, v in hits.items() if len(v) >= MIN_LEAVES}
    if not found:
        return NAMES[HEALTHY], {}
    order = sorted(found, key=lambda k: -sum(found[k]))
    return NAMES[order[0]], {NAMES[k]: max(found[k]) for k in order}


def draw(im, boxes, P, listed, path):
    im = im.copy()
    d = ImageDraw.Draw(im)
    lw = max(3, im.size[0] // 300)
    for (bc, part, (x0, y0, x1, y1)), p in zip(boxes, P):
        k = int(p.argmax())
        col = COLOR.get(NAMES[k], (0, 120, 255))
        d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        d.text((x0 + lw, y0 + lw), f"{PARTS.get(part, 'leaf')}: {NAMES[k]} {p[k]:.0%}", fill=col)
    head = "HEALTHY" if not listed else "DISEASED: " + ", ".join(f"{n} {c:.0%}" for n, c in listed.items())
    d.text((10, 10), head, fill=(255, 255, 255))
    im.save(path, quality=90)


target = next((a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--yolo", "--cfg")), "jetson_test_photos")
files = ([os.path.join(target, f) for f in sorted(os.listdir(target)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
         if os.path.isdir(target) else [target])
truth = {}
lab = os.path.join(target, "labels.csv") if os.path.isdir(target) else ""
if lab and os.path.exists(lab):
    truth = {r["file"]: r["true_class"] for r in csv.DictReader(open(lab))}
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

warm = ImageOps.exif_transpose(Image.open(files[0])).convert("RGB")       # warm-up (first GPU run is slow)
classify(warm, find_boxes(warm))

right, listed_ok, n_sick, t_det, t_clf, t_all, rows = 0, 0, 0, [], [], [], []
for f in files:
    t0 = time.time()
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    t1 = time.time()
    boxes = find_boxes(im)
    if dev == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()
    P = classify(im, boxes)
    if dev == "cuda":
        torch.cuda.synchronize()
    t3 = time.time()
    main, listed = palm_answer(P)
    t_det.append((t2 - t1) * 1000); t_clf.append((t3 - t2) * 1000); t_all.append((t3 - t0) * 1000)
    name = os.path.basename(f)
    real = truth.get(name)
    mark = "" if real is None else ("  ✓" if real == main else f"  ✗ (real: {real})")
    right += real == main
    if real and real != NAMES[HEALTHY]:
        n_sick += 1
        listed_ok += real in listed
    head = "HEALTHY" if not listed else "DISEASED: " + ", ".join(f"{n} {c:.0%}" for n, c in listed.items())
    print(f"\n{name}: {len(boxes)} boxes -> {head}{mark}   ({t_all[-1]:.0f} ms)")
    for i, ((bc, part, _), p) in enumerate(zip(boxes, P)):
        k = int(p.argmax())
        print(f"   {PARTS.get(part, 'leaf'):8s} {i + 1:2d}: box {bc:4.0%} | {NAMES[k]:20s} {p[k]:5.1%}")
    draw(im, boxes, P, listed, os.path.join(OUT, os.path.splitext(name)[0] + "_result.jpg"))
    rows.append(dict(file=name, boxes=len(boxes), main=main, diseases=";".join(f"{n}:{c:.2f}" for n, c in listed.items()) or "healthy",
                     true=real or "", ms=round(t_all[-1], 1)))

with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"\n{len(files)} photos | average per photo: YOLO {np.mean(t_det):.0f} ms + EdgeNeXt {np.mean(t_clf):.0f} ms"
      f" | total {np.mean(t_all):.0f} ms (including opening the photo)")
if truth:
    print(f"Correct (main answer): {right} / {len(files)} = {right / len(files):.1%}")
    if n_sick:
        print(f"Sick palms whose real disease was listed: {listed_ok} / {n_sick} = {listed_ok / n_sick:.1%}")
print("Pictures with boxes + results.csv saved in:", OUT)
