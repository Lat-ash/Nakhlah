"""Nakhlah on the drone: record video from the USB camera, and every time the RC switch is flipped, save the current
frame as a photo with GPS / altitude / time from the Pixhawk and classify it (YOLO26s -> EdgeNeXt, the best system).

SWITCH (--switch-mode, default "video"):
    video : flip the switch to HIGH -> start a video clip; flip back to LOW -> stop it (the clip is closed and safe).
            While a clip records, a photo is taken every --photo-every seconds (default 2) and classified with GPS.
    photo : the old way - the whole time is recorded, and every flip of the switch takes one photo.

Run (inside ~/Desktop/nakhlah_edgenet, with the env active):
    python3 nakhlah_capture.py                                   # USB camera 0, Pixhawk on /dev/ttyTHS1 at 921600, RC channel 7
    python3 nakhlah_capture.py --channel 8 --baud 57600          # other switch / baud rate
    python3 nakhlah_capture.py --no-mavlink                      # bench test without the Pixhawk: press Enter to take a photo
    python3 nakhlah_capture.py --no-classify                     # only record + take photos (classify later with nakhlah_pipeline.py)
Stop with Ctrl+C. Everything is saved in ~/nakhlah_flights/<date_time>/ :
    video_001.mp4, video_002.mp4 ... (1-minute pieces), photos/photo_001.jpg ..., records.csv (photo, time, GPS, altitude, answer, confidence per disease)
"""
import argparse, collections, csv, json, os, queue, sys, threading, time
from datetime import datetime

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--camera", default="0", help="camera index (0) or a video file / device path")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--no-video", action="store_true", help="do not save video (photos only)")
ap.add_argument("--switch-mode", choices=["video", "photo"], default="video", help="video: the switch starts/stops a clip | photo: each flip = 1 photo")
ap.add_argument("--photo-every", type=float, default=2, help="video mode: take + classify a photo every N seconds while a clip records (0 = off)")
ap.add_argument("--segment", type=float, default=60, help="start a new video file every N seconds (pulling the battery loses at most the last piece)")
ap.add_argument("--port", default="/dev/ttyTHS1", help="Pixhawk TELEM2 serial port on the Jetson (40-pin UART = /dev/ttyTHS1)")
ap.add_argument("--baud", type=int, default=921600, help="must match SERIAL2_BAUD on the Pixhawk (921 -> 921600, 57 -> 57600)")
ap.add_argument("--channel", type=int, default=7, help="RC channel of the free switch")
ap.add_argument("--no-mavlink", action="store_true", help="no Pixhawk: press Enter in the terminal to take a photo")
ap.add_argument("--auto-every", type=float, default=0, help="test mode: take a photo every N seconds")
ap.add_argument("--max-seconds", type=float, default=0, help="test mode: stop after N seconds")
ap.add_argument("--no-classify", action="store_true")
ap.add_argument("--out", default=os.path.expanduser("~/nakhlah_flights"))
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(args.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(os.path.join(RUN, "photos"), exist_ok=True)
stop = threading.Event()
state = dict(lat=None, lon=None, alt_m=None, rel_alt_m=None, sats=None, fix=None, rc=None)
triggers = queue.Queue()

# ------------------------------------------------------------------ 1. camera
cam = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera, cv2.CAP_V4L2 if args.camera.isdigit() else cv2.CAP_ANY)
if args.camera.isdigit():
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))     # MJPG lets USB cameras give full HD at 30 fps
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cam.set(cv2.CAP_PROP_FPS, args.fps)
ok, frame = cam.read()
if not ok:
    sys.exit(f"❌ Camera {args.camera} gives no picture. Check the USB cable, then run:  v4l2-ctl --list-devices")
H, W = frame.shape[:2]
fps = cam.get(cv2.CAP_PROP_FPS) or args.fps
print(f"📷 Camera ready: {W}x{H} at {fps:.0f} fps")
# the video is saved in 1-minute pieces, so pulling the battery loses at most the last minute
#   photo mode: video_001.mp4, video_002.mp4 ...      video mode: clip_001_part01.mp4, clip_001_part02.mp4 ... (one clip per switch flip)
SEG = dict(n=0, t=0.0, clip=0)
def new_video():
    global writer
    if writer is not None:
        writer.release()
    SEG["n"] += 1; SEG["t"] = time.time()
    name = f"video_{SEG['n']:03d}.mp4" if args.switch_mode == "photo" else f"clip_{SEG['clip']:03d}_part{SEG['n']:02d}.mp4"
    writer = cv2.VideoWriter(os.path.join(RUN, name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

def start_clip():
    SEG["clip"] += 1; SEG["n"] = 0
    new_video()
    print(f"🔴 RECORDING clip {SEG['clip']} ...", flush=True)

def stop_clip():
    global writer
    if writer is not None:
        writer.release(); writer = None
        print(f"⏹️  Clip {SEG['clip']} stopped and saved.", flush=True)

writer = None
if not args.no_video and args.switch_mode == "photo":
    new_video()
latest = dict(frame=frame)

# ------------------------------------------------------------------ 2. Pixhawk (MAVLink): RC switch + GPS
def mavlink_loop():
    from pymavlink import mavutil
    print(f"🔌 Connecting to the Pixhawk on {args.port} at {args.baud} baud ...")
    m = mavutil.mavlink_connection(args.port, baud=args.baud, source_system=42)
    while not stop.is_set() and not m.wait_heartbeat(timeout=10):   # keep waiting: the Pixhawk may be switched on later
        print("⏳ No heartbeat from the Pixhawk yet (is the drone battery in? TX/RX/GND wires? SERIAL2_PROTOCOL=2, SERIAL2_BAUD?) "
              "- still recording video, trying again ...", flush=True)
    if stop.is_set():
        return
    print(f"✅ Pixhawk connected (system {m.target_system}). " + (f"Switch on channel {args.channel}: HIGH = record, LOW = stop."
          if args.switch_mode == "video" else f"Flip the switch on channel {args.channel} to take a photo."), flush=True)
    for msg_id in (65, 33, 24):                         # RC_CHANNELS, GLOBAL_POSITION_INT, GPS_RAW_INT at 10 Hz
        m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                msg_id, 100000, 0, 0, 0, 0, 0)
    last_pos = None
    while not stop.is_set():
        msg = m.recv_match(type=["RC_CHANNELS", "GLOBAL_POSITION_INT", "GPS_RAW_INT"], blocking=True, timeout=1)
        if msg is None:
            continue
        t = msg.get_type()
        if t == "RC_CHANNELS":
            pwm = getattr(msg, f"chan{args.channel}_raw", 0)
            state["rc"] = pwm
            pos = "high" if pwm > 1700 else ("low" if 0 < pwm < 1300 else "mid")
            if args.switch_mode == "video":
                if pos != last_pos and pos in ("high", "low"):                         # also on the first reading
                    triggers.put("start" if pos == "high" else "stop")
            elif last_pos is not None and pos != last_pos and pos in ("high", "low"):   # every flip = one photo
                triggers.put(f"switch {pos} ({pwm})")
            last_pos = pos
        elif t == "GLOBAL_POSITION_INT":
            state.update(lat=msg.lat / 1e7, lon=msg.lon / 1e7, alt_m=msg.alt / 1000, rel_alt_m=msg.relative_alt / 1000)
        elif t == "GPS_RAW_INT":
            state.update(sats=msg.satellites_visible, fix=msg.fix_type)


def keyboard_loop():
    print("⌨️  Bench mode: press Enter to " + ("start / stop a clip" if args.switch_mode == "video" else "take a photo") + ", Ctrl+C to stop.")
    on = False
    while not stop.is_set():
        try:
            input()
            if args.switch_mode == "video":
                on = not on
                triggers.put("start" if on else "stop")
            else:
                triggers.put("keyboard")
        except EOFError:
            return


if args.no_mavlink:
    threading.Thread(target=keyboard_loop, daemon=True).start()
else:
    threading.Thread(target=mavlink_loop, daemon=True).start()

# ------------------------------------------------------------------ 3. the best system (YOLO26s run 1 + final EdgeNeXt)
classify_q = queue.Queue()
if not args.no_classify:
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

    def classify_loop():
        while True:
            try:
                rec, bgr = classify_q.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.time()
            ans, conf, keep, P = analyse(bgr)
            rec.update(answer=ans, confidence=round(conf, 3), leaves=len(keep), ms=round(1000 * (time.time() - t0)))
            out = bgr.copy()                                   # picture with the boxes, for the laptop / report
            for (bc, (x0, y0, x1, y1)), p in zip(keep, P):
                k = int(p.argmax())
                cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0) if NAMES[k] == "healthy" else (0, 0, 255), 3)
                cv2.putText(out, f"{NAMES[k]} {p[k]:.0%}", (int(x0), max(20, int(y0) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(rec["file"].replace(".jpg", "_result.jpg"), out)
            write_record(rec)
            print(f"   🌴 {os.path.basename(rec['file'])}: {ans} ({conf:.0%}, {len(keep)} leaves, {rec['ms']} ms)")
            classify_q.task_done()

    threading.Thread(target=classify_loop, daemon=True).start()

# ------------------------------------------------------------------ 4. records
FIELDS = ["photo", "file", "time", "trigger", "lat", "lon", "alt_m", "rel_alt_m", "gps_sats", "gps_fix", "gps_ok",
          "answer", "confidence", "leaves", "ms"]
REC = os.path.join(RUN, "records.csv")
lock = threading.Lock()
with open(REC, "w", newline="") as fh:
    csv.DictWriter(fh, fieldnames=FIELDS).writeheader()


def write_record(rec):
    with lock, open(REC, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore").writerow(rec)


# ------------------------------------------------------------------ 5. main loop: record video, take photos on triggers
n, t_start, t_auto = 0, time.time(), time.time()
print(f"🎬 Recording to {RUN}  (Ctrl+C to stop)")
try:
    while not stop.is_set():
        ok, frame = cam.read()
        if not ok:
            if not args.camera.isdigit():
                break                                          # end of a test video file
            print("⚠️ camera frame lost"); time.sleep(0.05); continue
        if writer is not None:
            if time.time() - SEG["t"] >= args.segment:
                new_video()                                    # close this piece (now safe) and start the next one
            writer.write(frame)
        if args.auto_every and time.time() - t_auto >= args.auto_every:
            triggers.put("auto"); t_auto = time.time()
        if args.switch_mode == "video" and writer is not None and args.photo_every and time.time() - t_auto >= args.photo_every:
            triggers.put(f"clip {SEG['clip']}"); t_auto = time.time()
        while not triggers.empty():
            why = triggers.get()
            if why == "start":
                if writer is None and not args.no_video:
                    start_clip(); t_auto = time.time() - args.photo_every   # first photo right away
                continue
            if why == "stop":
                stop_clip()
                continue
            n += 1
            path = os.path.join(RUN, "photos", f"photo_{n:03d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gps_ok = state["fix"] is not None and state["fix"] >= 3 and state["lat"] not in (None, 0)
            rec = dict(photo=n, file=path, time=datetime.now().isoformat(timespec="seconds"), trigger=why,
                       lat=state["lat"], lon=state["lon"], alt_m=state["alt_m"], rel_alt_m=state["rel_alt_m"],
                       gps_sats=state["sats"], gps_fix=state["fix"], gps_ok=gps_ok)
            print(f"📸 photo {n} ({why}) | GPS {'%.6f, %.6f' % (state['lat'], state['lon']) if gps_ok else 'not available'}")
            if args.no_classify:
                write_record(rec)
            else:
                classify_q.put((rec, frame.copy()))
        if args.max_seconds and time.time() - t_start >= args.max_seconds:
            break
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    time.sleep(0.2)
    if not args.no_classify:
        print("…finishing the photos still being classified")
        classify_q.join()                                      # wait until every photo has its result
    cam.release()
    if writer is not None:
        writer.release()
    print(f"\n✅ Stopped. {SEG['clip']} clips | {n} photos | everything saved in {RUN}")
