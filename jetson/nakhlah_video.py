"""Nakhlah: run the best system (YOLO26s run 1 -> final EdgeNeXt) on a VIDEO FILE and see what it detects.

Run (inside ~/Desktop/nakhlah_edgenet, with the env active):
    python3 nakhlah_video.py my_leaf_video.mp4               # analyse 1 frame every 0.5 s of video
    python3 nakhlah_video.py my_leaf_video.mp4 --every 1      # 1 frame every 1 s (faster)
    python3 nakhlah_video.py my_leaf_video.mp4 --show         # also show the result live in a window (press q to stop)

Saves in ~/nakhlah_videos/<video name>_<date_time>/ :
    result.mp4      the video with YOLO's boxes and EdgeNeXt's disease + confidence on each leaf
    frames/         the analysed frames as pictures (with boxes)
    results.csv     time in the video, answer for the frame, confidence, number of leaves, ms
and prints a summary (how often each class was the answer).
"""
import argparse, collections, csv, json, os, sys, time
from datetime import datetime

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video", help="path to the video file (mp4, mov, avi ...)")
ap.add_argument("--every", type=float, default=0.5, help="analyse one frame every N seconds of video")
ap.add_argument("--show", action="store_true", help="show the result in a window while it runs (needs a screen)")
ap.add_argument("--out", default=os.path.expanduser("~/nakhlah_videos"))
args = ap.parse_args()

if not os.path.exists(args.video):
    sys.exit(f"❌ Video not found: {args.video}")
HERE = os.path.dirname(os.path.abspath(__file__))
name = os.path.splitext(os.path.basename(args.video))[0]
RUN = os.path.join(args.out, f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
os.makedirs(os.path.join(RUN, "frames"), exist_ok=True)

# ------------------------------------------------------------------ the best system (same as the drone program)
import timm, torch
from PIL import Image
from ultralytics import YOLO
dev = "cuda" if torch.cuda.is_available() else "cpu"
Y = json.load(open(os.path.join(HERE, "nakhlah_yolo.json")))
M = json.load(open(os.path.join(HERE, "nakhlah_model.json")))
det = YOLO(os.path.join(HERE, "nakhlah_yolo26s.pt"))
clf = timm.create_model(M["timm_name"].split(".")[0], pretrained=False, num_classes=M["num_classes"])
clf.load_state_dict(torch.load(os.path.join(HERE, "edgenext_nakhlah.pt"), map_location="cpu"))
clf = clf.to(dev).eval()
NAMES, SIZE, PAD, MIN_PX, CONF = M["class_names"], M["image_size"], Y["crop_padding"], Y["min_box_px"], Y["conf"]
MEAN, STD = np.array(M["mean"], np.float32), np.array(M["std"], np.float32)
print(f"🧠 Models loaded on {dev.upper()} | YOLO {Y['imgsz']} px, conf {CONF} + fallback | majority vote")


@torch.no_grad()
def analyse(bgr):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    r = det.predict(bgr, imgsz=Y["imgsz"], conf=0.01, max_det=20, verbose=False)[0]
    bx = sorted(zip(r.boxes.conf.cpu().tolist(), r.boxes.xyxy.cpu().tolist()), key=lambda b: -b[0])
    keep = [b for b in bx if b[0] >= CONF and b[1][2] - b[1][0] >= MIN_PX and b[1][3] - b[1][1] >= MIN_PX] or bx[:1]
    if not keep:
        return "no leaf found", 0.0, [], []
    Wd, Hd = im.size
    xs = []
    for _, (x0, y0, x1, y1) in keep:
        w, h = x1 - x0, y1 - y0
        c = im.crop((int(max(0, x0 - PAD * w)), int(max(0, y0 - PAD * h)), int(min(Wd, x1 + PAD * w)), int(min(Hd, y1 + PAD * h))))
        xs.append((np.asarray(c.resize((SIZE, SIZE)), np.float32) / 255.0 - MEAN) / STD)
    P = torch.softmax(clf(torch.from_numpy(np.stack(xs).transpose(0, 3, 1, 2)).to(dev)).float(), 1).cpu().numpy()
    k = collections.Counter(P.argmax(1).tolist()).most_common(1)[0][0]
    conf = float(np.mean([p[k] for p in P if p.argmax() == k]))
    return NAMES[k], conf, keep, P


def draw(bgr, ans, conf, keep, P):
    out = bgr.copy()
    th = max(2, out.shape[1] // 600)
    for (bc, (x0, y0, x1, y1)), p in zip(keep, P):
        k = int(p.argmax())
        col = (0, 200, 0) if NAMES[k] == "healthy" else (0, 0, 255)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), col, th)
        cv2.putText(out, f"{NAMES[k]} {p[k]:.0%}", (int(x0), max(25, int(y0) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, (255, 255, 255), th)
    cv2.rectangle(out, (0, 0), (out.shape[1], 45 * th // 2), (0, 0, 0), -1)
    cv2.putText(out, f"PALM: {ans} ({conf:.0%}, {len(keep)} leaves)", (10, 16 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * th, (255, 255, 255), th)
    return out


# ------------------------------------------------------------------ go through the video
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    sys.exit(f"❌ Could not open the video: {args.video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
step = max(1, round(args.every * fps))
print(f"🎬 {args.video}: {W}x{H}, {fps:.0f} fps, {total / fps:.0f} s | analysing 1 frame every {args.every} s")
writer = cv2.VideoWriter(os.path.join(RUN, "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
rows, last, i = [], None, 0
with open(os.path.join(RUN, "results.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=["second", "answer", "confidence", "leaves", "ms"])
    wr.writeheader()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                t0 = time.time()
                ans, conf, keep, P = analyse(frame)
                ms = round(1000 * (time.time() - t0))
                last = (ans, conf, keep, P)
                sec = round(i / fps, 1)
                row = dict(second=sec, answer=ans, confidence=round(conf, 3), leaves=len(keep), ms=ms)
                wr.writerow(row); rows.append(row)
                shown = draw(frame, *last)
                cv2.imwrite(os.path.join(RUN, "frames", f"t{sec:07.1f}s.jpg"), shown)
                print(f"   {sec:6.1f} s: {ans} ({conf:.0%}, {len(keep)} leaves, {ms} ms)")
            else:
                shown = draw(frame, *last) if last else frame       # keep the last boxes until the next analysis
            writer.write(shown)
            if args.show:
                cv2.imshow("Nakhlah", cv2.resize(shown, (min(W, 1280), int(H * min(W, 1280) / W))))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            i += 1
    except KeyboardInterrupt:
        print("…stopped")
cap.release(); writer.release()
if args.show:
    cv2.destroyAllWindows()

if rows:
    cnt = collections.Counter(r["answer"] for r in rows)
    print(f"\n========== SUMMARY: {len(rows)} frames analysed ==========")
    for k, v in cnt.most_common():
        print(f"   {k:<20} {v:4d} frames ({v / len(rows):.0%})")
    print(f"   most common answer: {cnt.most_common(1)[0][0]} | average {np.mean([r['ms'] for r in rows[1:]] or [0]):.0f} ms per frame")
print(f"\n✅ Saved in {RUN}  (result.mp4, frames/, results.csv)")
