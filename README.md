# Nakhlah – AI-Powered Drone for Date Palm Health Monitoring

Senior project (CS499 / SE499), Prince Sultan University, 2026.
Team: Fatoon Aljunobi, Latifah Bin Showaish, Lubna Alnajran, Rama Hassan, Shiyam Alhudaithy.
Supervisor: Dr. Fatimah Alshannaq.

Nakhlah is a quadcopter with an NVIDIA Jetson Orin Nano on board. While the drone flies, the pilot starts and
stops video recording with a switch on the remote control. During recording, a photo is taken every 2 seconds,
the **YOLO26s** leaf detector finds the leaf regions, and the **EdgeNeXt-Base** classifier labels each leaf as one
of four classes (healthy, fungal spotting, fungal wilt/blight, nutrient deficiency). Every result is saved with its GPS
position from the Pixhawk flight controller.

## Results

| | Result |
|---|---|
| EdgeNeXt-Base, locked test set (hand-drawn boxes) | crop macro-F1 0.835, whole photo 0.855 |
| Full pipeline (YOLO26s + EdgeNeXt), 270 test photos | 71.5% accuracy, macro-F1 0.713 |
| Speed on the Jetson | 40.8 ms per leaf crop, about 0.2 s per photo |
| Full system flight | 7 November 2026, Jetson and camera on board |

## Repository

```
notebooks/
  1_classifier_fair_tests/   EdgeNeXt-Base on the four fair tests (same split, crops and recipe as the other models)
    1_EdgeNeXt_SamePlaces_MAT_to_MAT.ipynb          train 85% of Ammariyah+Thumamah+Madinah -> locked 15%
    2_EdgeNeXt_NewFarm_Ammariyah_to_Thumamah.ipynb  never-seen farm
    3_EdgeNeXt_NewRegion_AT_to_Madinah.ipynb        never-seen region (no Qassim)
    4_EdgeNeXt_NewRegion_ATQ_to_Madinah.ipynb       never-seen region (with Qassim in training)
  2_final_classifier/
    Nakhlah_EdgeNeXt_FINAL_Model_for_Jetson.ipynb   trains the deployed model, exports edgenext_nakhlah.pt + nakhlah_model.json
  3_leaf_detector/
    Nakhlah_YOLO26s_640_Leaf_Detector.ipynb         trains YOLO26s (one "leaf" class), tests the full pipeline, exports nakhlah_yolo26s.pt + nakhlah_yolo.json
jetson/
  nakhlah_classify.py  classifies ready-made leaf crops (speed and accuracy test of EdgeNeXt)
  nakhlah_pipeline.py  full pipeline on photos: YOLO26s -> crops -> EdgeNeXt -> majority vote
  nakhlah_capture.py   flight program: camera + MAVLink (GPS, switch) + classification, saves clips, photos and records.csv
  nakhlah_video.py     runs the pipeline on a video file and draws the results on it
  nakhlah.service      starts nakhlah_capture.py automatically when the Jetson is powered on
```

## Running the notebooks (Google Colab)

1. Runtime → Change runtime type → **T4 GPU** (or A100).
2. Paste your **Roboflow API key** in cell 1 (replace `PASTE_YOUR_ROBOFLOW_API_KEY_HERE`). Never commit a real key.
3. Run all. Every notebook rebuilds the frozen split (`Nakhlah_ATM_combined_5folds` in Google Drive) and **stops if it
   does not match**, so all experiments use exactly the same data.

## Running on the Jetson

Environment: JetPack 6.2.1, Python venv, PyTorch 2.11.0 (NVIDIA Jetson GPU build), cuDSS 0.7.1, timm, pymavlink,
pyserial, OpenCV < 4.12, NumPy < 2. Install Ultralytics with `pip install --no-deps ultralytics`, otherwise it replaces
the GPU build of PyTorch with a CPU-only one.

Put these files from the notebooks in the same folder as the programs:
`edgenext_nakhlah.pt`, `nakhlah_model.json`, `nakhlah_yolo26s.pt`, `nakhlah_yolo.json`.

```bash
# bench test without the drone (press Enter to start/stop a clip)
python3 nakhlah_capture.py --no-mavlink
# on the drone (Pixhawk TELEM2 on /dev/ttyTHS1, switch SwB = channel 6)
python3 nakhlah_capture.py --channel 6
# analyse a recorded video
python3 nakhlah_video.py ~/nakhlah_flights/<folder>/clip_001_part01.mp4
```

Results are saved in `~/nakhlah_flights/<date_time>/`: `clip_XXX_partYY.mp4`, `photos/`, `records.csv`
(time, GPS, altitude, class, confidence).

Model weights and datasets are not in this repository (they are kept in the team's Google Drive).
