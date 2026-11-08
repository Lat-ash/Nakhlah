# Nakhlah: AI-Powered Drone for Date Palm Health Monitoring

**Nakhlah** (نخلة, "palm tree") is a CS499 / SE499 senior project at Prince Sultan University, 2026. A quadcopter carries a camera and an NVIDIA Jetson Orin Nano. While the drone flies, the Jetson takes a photo every 2 seconds, finds the leaves in it, classifies each leaf into one of four health classes, and saves the result with its GPS position.

This repository holds the **AI and on-board software**: the Google Colab notebooks that train and test the models, and the programs that run on the Jetson.

| Class | Meaning |
|---|---|
| `fungalSpotting` | Fungal leaf spots |
| `fungalWiltBlight` | Fungal wilt / blight |
| `healthy` | Healthy leaf |
| `nutrientDeficiency` | Nutrient deficiency |

> **The task is classification.** YOLO26s only finds the leaf regions so they can be cropped. EdgeNeXt-Base then decides the class of each leaf. The project does not deliver segmentation or severity estimation.

---

## Contents

- [Repository layout](#repository-layout)
- [How it works](#how-it-works)
- [Data](#data)
- [Method](#method)
- [Results](#results)
- [Running the notebooks](#running-the-notebooks)
- [Running on the Jetson](#running-on-the-jetson)
- [Limitations and future work](#limitations-and-future-work)
- [Team](#team)

---

## Repository layout

```
Nakhlah/
├── notebooks/
│   ├── 1_classifier_fair_tests/                       # same split, crops and recipe as the other models
│   │   ├── 1_EdgeNeXt_SamePlaces_MAT_to_MAT.ipynb      # 85% of Ammariyah + Thumamah + Madinah -> locked 15%
│   │   ├── 2_EdgeNeXt_NewFarm_Ammariyah_to_Thumamah.ipynb   # a farm never seen
│   │   ├── 3_EdgeNeXt_NewRegion_AT_to_Madinah.ipynb    # a region never seen (no Qassim)
│   │   └── 4_EdgeNeXt_NewRegion_ATQ_to_Madinah.ipynb   # a region never seen (Qassim in training)
│   ├── 2_final_classifier/
│   │   └── Nakhlah_EdgeNeXt_FINAL_Model_for_Jetson.ipynb    # trains the deployed classifier, exports .pt + .json
│   └── 3_leaf_detector/
│       └── Nakhlah_YOLO26s_640_Leaf_Detector.ipynb     # trains YOLO26s ("leaf"), tests the full pipeline, exports .pt + .json
├── jetson/
│   ├── nakhlah_capture.py      # flight program: camera + MAVLink (GPS, RC switch) + classification
│   ├── nakhlah_pipeline.py     # full pipeline on a folder of photos
│   ├── nakhlah_classify.py     # classifier only, on ready-made leaf crops (speed + accuracy test)
│   ├── nakhlah_video.py        # runs the pipeline on a recorded video and draws the results
│   └── nakhlah.service         # starts nakhlah_capture.py when the Jetson powers on
├── .gitignore
└── README.md
```

## How it works

```
 Pilot flips switch SwB on the remote ──► Pixhawk 2.4.8 ──(MAVLink, TELEM2 → /dev/ttyTHS1)──► Jetson Orin Nano
                                          + M8N GPS                                              │
 ELP AR0234 USB camera ──────────────────────────────────────────────────────────────────────────┤
                                                                                                  ▼
                       record video  ─►  every 2 s: photo  ─►  YOLO26s finds leaves  ─►  crop each leaf (224×224)
                                                                                                  │
                                       records.csv  ◄──  photo answer (majority vote)  ◄──  EdgeNeXt-Base per leaf
                              (time, GPS, altitude, class, confidence)
```

Everything is saved on the Jetson in `~/nakhlah_flights/<date_time>/`. After landing, the folder is copied to the laptop, where the farmer reviews the results.

## Data

| Source | Photos with boxes | Role |
|---|---|---|
| Al-Ammariyah (own collection, Roboflow `alamariah` v1) | 1,014 | train + locked test |
| Al-Thumamah (own collection, Roboflow `althumamah` v1) | 1,027 | train + locked test |
| Al-Madinah, Namoun et al. (2024), Roboflow `combined-dataset-for-naklah` v3 | 485 after cleaning | external test, or train + locked test |
| Al-Qassim mini-set (same Roboflow project) | 80 after cleaning | training only, never tested |

Own data per class (Al-Ammariyah + Al-Thumamah, 2,041 photos):

| fungalSpotting | fungalWiltBlight | healthy | nutrientDeficiency |
|---:|---:|---:|---:|
| 544 | 464 | 693 | 340 |

Cleaning steps in every notebook:
- Exact duplicate photos are found by MD5 hash and only one copy is kept. If the same photo was saved under two different classes, both copies are removed.
- The Madinah `insectInfestation` class is not one of our four classes, so those boxes are skipped.

Our two farms can be downloaded from the [`data-v1` release](https://github.com/Lat-ash/Nakhlah/releases/tag/data-v1). The notebooks download all data from Roboflow at run time.

## Method

**Split (frozen, reused by every notebook)**
- 15% of each place is a **locked test** set.
- The remaining 85% is divided into **5 folds** with `StratifiedGroupKFold` (seed 42), balanced by place and class.
- **Leaf groups** stop near-identical photos from landing in both training and test. For our farms, a group is a block of 10 neighbouring image numbers (e.g. `IMG_9190`–`IMG_9199`). This is a heuristic and has not been checked against the physical leaves. Madinah file names carry a leaf number, which is used directly.
- Every notebook rebuilds the split, compares it with the frozen copy in Google Drive (`Nakhlah_ATM_combined_5folds`), and **stops if anything changed**.

**Classifier (EdgeNeXt-Base)**
- `timm` `edgenext_base.in21k_ft_in1k` with a new 4-class head (~18M parameters)
- Leaf crops with 10% padding, resized to 224×224; boxes under 20 px are skipped
- 5 epochs training only the head (lr 1e-3), then 10 epochs fine-tuning the whole model (lr 1e-4)
- Batch 32, class-weighted cross-entropy, ImageNet normalisation, seed 42
- Online augmentation on training crops only: flips, 90° rotations, light brightness/contrast and saturation/value changes, occasional motion blur. Hue is not changed, so disease colours are kept.

**Leaf detector (YOLO26s)**
- One class, `leaf`: all four disease labels are merged
- 640 px, up to 100 epochs with early stopping, validation on fold 0
- Confidence threshold 0.10; if no box passes, the single most confident box is used

**Metrics**: accuracy, balanced accuracy, macro precision, macro recall, **macro F1**, per-class scores and confusion matrices, per crop and per whole photo (majority vote over the photo's leaves).

## Results

### Summary

| | Result |
|---|---|
| EdgeNeXt-Base, locked test set (hand-drawn boxes) | crop macro-F1 0.835, whole photo 0.855 |
| Full pipeline (YOLO26s + EdgeNeXt), 270 test photos | 71.5% accuracy, macro-F1 0.713 |
| Speed on the Jetson | 40.8 ms per leaf crop, about 0.2 s per photo |
| Full system flight | 7 November 2026, Jetson and camera on board |

### Generalisation to a new region: train on our farms (+ Qassim), test on all of Madinah

The Madinah test is the same in every run: 665 crops from 484 photos, never used for training. Values are crop macro F1, without / with augmentation.

| Model | Params | No aug | Aug |
|---|---:|---:|---:|
| **EdgeNeXt-Base** | 18M | **0.643** | **0.642** |
| BioCLIP | — | 0.638 | 0.634 |
| TinyViT-11M | 11M | 0.617 | 0.623 |
| EfficientNetV2-S | — | 0.599 | 0.621 |
| ConvNeXt-Tiny | 28M | 0.610 | 0.614 |

Without Qassim, EdgeNeXt-Base scored 0.638 / 0.633 on the same test.

### Generalisation to a new farm: train on Al-Ammariyah, test on all of Al-Thumamah

Test set: 2,192 crops from 1,027 photos.

| Setup | Level | Accuracy | Balanced acc. | Macro F1 |
|---|---|---:|---:|---:|
| No aug | crop | 0.848 | 0.794 | 0.791 |
| Aug | crop | 0.867 | 0.823 | **0.821** |
| Aug | whole photo | 0.864 | 0.808 | 0.811 |

Most errors come from confusing **nutrientDeficiency with fungalWiltBlight**.

### Why EdgeNeXt-Base

It is the best or tied-best model on the external Madinah test, it is much smaller than ConvNeXt-Tiny (18M vs 28M parameters), and it is designed for edge devices like the Jetson.

## Running the notebooks

1. Open a notebook in Google Colab and choose **Runtime → Change runtime type → T4 GPU** (or A100).
2. In cell 1, replace `PASTE_YOUR_ROBOFLOW_API_KEY_HERE` with your Roboflow API key. **Never commit a real key.**
3. **Run all.** Check that the split cells print `✅ Same split as the frozen one`.

**Order**: the fair tests in `1_classifier_fair_tests/` (any order) → `2_final_classifier/` → `3_leaf_detector/` (needs the final classifier's Drive folder).

The notebooks export the four model files the Jetson needs: `edgenext_nakhlah.pt`, `nakhlah_model.json`, `nakhlah_yolo26s.pt` and `nakhlah_yolo.json`. Model weights are not in this repository; they are kept in the team's Google Drive.

## Running on the Jetson

**Environment**: JetPack 6.2.1, Python venv, PyTorch 2.11.0 (NVIDIA Jetson GPU build), cuDSS 0.7.1, timm, pymavlink, pyserial, OpenCV < 4.12, NumPy < 2. Install Ultralytics with `pip install --no-deps ultralytics`, otherwise it replaces the GPU build of PyTorch with a CPU-only one.

Put the four model files in the same folder as the programs, then:

```bash
# bench test without the drone (press Enter to start/stop a clip)
python3 nakhlah_capture.py --no-mavlink

# on the drone (Pixhawk TELEM2 on /dev/ttyTHS1, switch SwB = channel 6)
python3 nakhlah_capture.py --channel 6

# analyse a recorded video
python3 nakhlah_video.py ~/nakhlah_flights/<folder>/clip_001_part01.mp4
```

Each flight is saved in `~/nakhlah_flights/<date_time>/`:

| File | Contents |
|---|---|
| `clip_XXX_partYY.mp4` | video, saved in 1-minute pieces so a pulled battery loses at most the last minute |
| `photos/` | the photos that were classified |
| `records.csv` | time, GPS, altitude, class, confidence and number of leaves for each photo |

**Auto-start**: copy `jetson/nakhlah.service` to `/etc/systemd/system/`, then run `sudo systemctl daemon-reload && sudo systemctl enable --now nakhlah`. The capture program then starts by itself when the drone battery powers the Jetson.

## Limitations and future work

- **Domain shift is the main open problem.** The classifier scores about 0.82 macro F1 on a new farm in the same region, but only about 0.64 on Madinah.
- **Automatic leaf finding costs accuracy.** With YOLO26s crops, whole-photo macro F1 is 0.713, versus 0.855 with hand-drawn boxes. The 0.10 threshold and fallback were added after the locked test had already been used once, so 0.713 is an optimistic estimate.
- **Leaf groups are a heuristic.** A block of 10 neighbouring image numbers is not guaranteed to be one physical leaf.
- **The Madinah numbers guided model choice**, so Madinah is an external development set, not an untouched final test.
- **Whole-photo majority vote** can be misleading when one photo contains leaves of more than one class.
- **Planned next steps**: improve the leaf detector, longer fine-tuning with validation-only early stopping, larger crops (320 px), and per-class Madinah error analysis.

## Team

CS499 / SE499 Senior Project, College of Computer and Information Sciences, Prince Sultan University, 2026.
Supervisor: **Dr. Fatimah Alshannaq**

- Fatoon Aljunobi
- Latifah Bin Showaish
- Lubna Alnajran
- Rama Hassan
- Shiyam Alhudaithy

## Acknowledgements

- Al-Madinah date palm leaf dataset: Namoun et al. (2024)
- [timm](https://github.com/huggingface/pytorch-image-models) for EdgeNeXt; EdgeNeXt by Maaz et al. (2022)
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO26, and [Roboflow](https://roboflow.com) for labelling and dataset hosting