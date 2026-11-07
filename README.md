# Nakhlah: AI for Date Palm Leaf Disease Classification

**Nakhlah** (نخلة, "palm tree") is a CS499 senior project at Prince Sultan University. A drone carries a camera and an NVIDIA Jetson Orin Nano. The Jetson classifies date palm leaves into four health classes during the flight and saves the results. After the flight, the farmer reviews those results in a desktop app.

This repository holds the **AI part** of the project: the Google Colab notebooks that build the data splits, compare models fairly, train the final classifier and export it for the Jetson.

| Class | Meaning |
|---|---|
| `fungalSpotting` | Fungal leaf spots |
| `fungalWiltBlight` | Fungal wilt / blight |
| `healthy` | Healthy leaf |
| `nutrientDeficiency` | Nutrient deficiency |

> **The task is classification.** Bounding boxes are only used to crop each leaf (the region of interest) before it is classified. The project does not deliver object detection, segmentation or severity estimation.

---

## Contents

- [Repository layout](#repository-layout)
- [Pipeline](#pipeline)
- [Data](#data)
- [Method](#method)
- [Results](#results)
- [How to run](#how-to-run)
- [Model files for the Jetson](#model-files-for-the-jetson)
- [Limitations and future work](#limitations-and-future-work)
- [Team](#team)

---

## Repository layout

```
Nakhlah/
├── notebooks/
│   ├── experiments/                                   # fair tests: same split, same recipe, only the model or data changes
│   │   ├── Nakhlah_EdgeNeXt_MAT_to_MAT.ipynb           # 85% of all 3 places -> locked 15%, and farms -> Madinah
│   │   ├── Nakhlah_EdgeNeXtBase_Ammariyah_to_Thumamah.ipynb   # one farm -> a farm never seen
│   │   ├── Nakhlah_EdgeNeXtBase_AT_to_Madinah_NoQassim.ipynb  # our farms -> Madinah, without Qassim
│   │   └── Nakhlah_EdgeNeXtBase_ATQ_to_Madinah.ipynb   # our farms + Qassim -> Madinah
│   └── final/
│       ├── Nakhlah_EdgeNeXt_FINAL_Model_for_Jetson.ipynb      # trains the final model once, tests it once, exports it
│       └── Nakhlah_YOLO26s_640_bestRunWithEdgeNet.ipynb       # experimental leaf-cropping stage (see Limitations)
├── models/README.md        # which model files exist and where they go (weights are not in git)
├── data/README.md          # where the data comes from (images are not in git)
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline

```
 Drone camera (ELP AR0234, global shutter)
        │  frames
        ▼
 Jetson Orin Nano Super ── leaf crop (box + 10% padding, 224×224) ── EdgeNeXt-Base ── class per leaf
        │                                                                     │
        │                                                   results saved to a folder during flight
        ▼                                                                     ▼
 Pixhawk 2.4.8 + M8N GPS (flight control)            after landing: folder copied to the laptop
                                                      → Nakhlah app (Dashboard · Records · Illness)
```

The desktop app only reads saved files. It does not stream video and never runs the model.

## Data

| Source | Photos with boxes | Role |
|---|---|---|
| Al-Ammariyah (own collection, Roboflow `alamariah` v1) | 1,014 | train + locked test |
| Al-Thumamah (own collection, Roboflow `althumamah` v1) | 1,027 | train + locked test |
| Al-Madinah, Namoun et al. (2024), from Roboflow `combined-dataset-for-naklah` v3 | 485 after cleaning | external test, or train + locked test |
| Al-Qassim mini-set (same Roboflow project) | 80 after cleaning | training only, never tested |

Own data per class (Al-Ammariyah + Al-Thumamah, 2,041 photos):

| fungalSpotting | fungalWiltBlight | healthy | nutrientDeficiency |
|---:|---:|---:|---:|
| 544 | 464 | 693 | 340 |

Cleaning steps in every notebook:
- Exact duplicate photos are found by MD5 hash and only one copy is kept. If the same photo was saved under two different classes, both copies are removed.
- The Madinah `insectInfestation` class is not one of our four classes, so those boxes are skipped.

Images are **not** stored in the git repository. Our two farms can be downloaded from the [`data-v1` release](https://github.com/Lat-ash/Nakhlah/releases/tag/data-v1). See [`data/README.md`](data/README.md) for details.

## Method

**Split (frozen, reused by every notebook)**
- 15% of each place is a **locked test** set. It was used only once, after the final model was chosen.
- The remaining 85% is divided into **5 folds** with `StratifiedGroupKFold` (seed 42), balanced by place and class.
- **Leaf groups** stop near-identical photos from landing in both training and test. For our farms, a group is a block of 10 neighbouring image numbers (e.g. `IMG_9190`–`IMG_9199`). This is a heuristic and has not been checked against the physical leaves. Madinah file names carry a leaf number, which is used directly.
- Every notebook rebuilds the split, compares it with the frozen `split.csv`/`roles.csv`, and **stops if anything changed**. It also checks that no group or photo hash appears in both training and test.

**Crops**: each labelled box is cropped with 10% padding and resized to 224×224. Boxes under 20 px are skipped.

**Training recipe (the same for every model)**
- Model: `timm` `edgenext_base.in21k_ft_in1k` with a new 4-class head (~18M parameters)
- 5 epochs training only the head (lr 1e-3), then 10 epochs fine-tuning the whole model (lr 1e-4)
- Batch 32, class-weighted cross-entropy, ImageNet normalisation, seed 42
- Online augmentation on training crops only: flips, 90° rotations, light brightness/contrast and saturation/value changes, occasional motion blur. Hue is not changed, so disease colours are kept.
- Each fair test trains 5 fold models, and the 5 models vote on the test set.

**Metrics**: accuracy, balanced accuracy, macro precision, macro recall, **macro F1**, per-class scores and confusion matrices. Each is reported per crop and per whole photo (majority vote over the photo's crops). Crop-level is the cleaner measure, because one photo can contain leaves of more than one class.

## Results

All numbers below come from the notebooks in this repository.

### Final model: locked 15% test (used once)

Trained on 85% of Al-Ammariyah + Al-Thumamah + Madinah, plus all of Qassim, with augmentation. Tested on the locked 15% of the three places (270 photos).

| Level | Accuracy | Macro F1 |
|---|---:|---:|
| Whole photo (labelled boxes) | 0.859 | 0.855 |

The crop-level metrics are written to `final_test_results.csv` by the final notebook.

### Generalisation to a new region: train on our farms (+ Qassim), test on all of Madinah

The Madinah test is the same in every run: 665 crops from 484 photos, never used for training. Values are crop macro F1, without / with augmentation.

| Model | Params | No aug | Aug |
|---|---:|---:|---:|
| **EdgeNeXt-Base** | 18M | **0.643** | **0.642** |
| BioCLIP | — | 0.638 | 0.634 |
| TinyViT-11M | 11M | 0.617 | 0.623 |
| EfficientNetV2-S | — | 0.599 | 0.621 |
| ConvNeXt-Tiny | 28M | 0.610 | 0.614 |

Without Qassim, EdgeNeXt-Base scored 0.638 / 0.633 on the same test, so adding Qassim gave only a small gain.

### Generalisation to a new farm: train on Al-Ammariyah, test on all of Al-Thumamah

Test set: 2,192 crops from 1,027 photos.

| Setup | Level | Accuracy | Balanced acc. | Macro F1 |
|---|---|---:|---:|---:|
| No aug | crop | 0.848 | 0.794 | 0.791 |
| Aug | crop | 0.867 | 0.823 | **0.821** |
| Aug | whole photo | 0.864 | 0.808 | 0.811 |

Per-class crop F1 with augmentation: fungalSpotting 0.904 · fungalWiltBlight 0.747 · healthy 0.943 · nutrientDeficiency 0.689. Most errors come from confusing **nutrientDeficiency with fungalWiltBlight**.

### Why EdgeNeXt-Base

It is the best or tied-best model on the external Madinah test. It is much smaller than ConvNeXt-Tiny (18M vs 28M parameters), and it is designed for edge devices like the Jetson. On a Colab T4 it takes about 9.5 ms per crop. The Jetson will be slower, and its speed has not been measured yet.

## How to run

The notebooks are written for **Google Colab** and save their outputs to Google Drive.

1. Open a notebook in Colab and choose **Runtime → Change runtime type → T4 GPU**.
2. In cell 1, replace `PASTE_YOUR_ROBOFLOW_API_KEY_HERE` with your Roboflow API key. **Never commit a real key.** A safer option is to save it in Colab Secrets and load it with `from google.colab import userdata; userdata.get("ROBOFLOW_KEY")`.
3. **Run all.** Before trusting any result, check that the split cells print `✅ Same split as the frozen one`.
4. If Colab disconnects, run all again. Folds that already finished are loaded from Drive and skipped.

**Run order**
1. `experiments/*`: the fair tests, in any order. They reuse the frozen split from the Drive folder `Nakhlah_ATM_combined_5folds`. If that folder is missing, the split is rebuilt with the same method and seed and saved.
2. `final/Nakhlah_EdgeNeXt_FINAL_Model_for_Jetson.ipynb`: trains the final model (~20–30 min) and also saves `qassim_split.csv`.
3. `final/Nakhlah_YOLO26s_640_bestRunWithEdgeNet.ipynb`: optional. It needs the final model's Drive folder.

Approximate run times on a T4: about 45–90 min for each single-test notebook, and 1.5–3 h for `MAT_to_MAT` (two tests).

To run locally or on the Jetson instead of Colab, install the packages in `requirements.txt`. On the Jetson, install PyTorch from NVIDIA's JetPack wheels rather than from PyPI.

## Model files for the Jetson

The final notebook writes these files to `MyDrive/Nakhlah_EdgeNeXt_FINAL_model/`:

| File | Purpose |
|---|---|
| `edgenext_nakhlah.pt` | trained weights (PyTorch `state_dict`) |
| `nakhlah_model.json` | class order, image size, padding and normalisation that inference must use |
| `edgenext_nakhlah.onnx` | the same model in ONNX format, for TensorRT later |
| `jetson_test_crops.zip` | 40 locked-test crops (10 per class) with labels, to check the model on the Jetson |
| `final_test_results.csv` | locked-test metrics |

The weights are not committed to git. See [`models/README.md`](models/README.md).

## Limitations and future work

- **Domain shift is the main open problem.** The model scores about 0.82 macro F1 on a new farm in the same region, but only about 0.64 on Madinah. Class meanings may also differ between sites, and this has not been checked yet.
- **Leaf groups are a heuristic.** A block of 10 neighbouring image numbers is not guaranteed to be one physical leaf.
- **The Madinah numbers guided model choice**, so Madinah is an external development set, not an untouched final test.
- **Whole-photo majority vote** can be misleading when one photo contains leaves of more than one class.
- **The automatic leaf-cropping stage is experimental.** The YOLO26s notebook trains a one-class `leaf` box model to replace the hand-drawn boxes. On the locked test it reached box mAP50 0.379. With its crops, the classifier reached whole-photo macro F1 0.670, versus 0.855 with hand-drawn boxes. A lower threshold (0.10, chosen on fold 0) plus a "use the most confident box" fallback raised this to 0.713. That second result was measured after the locked test had already been used once, so treat it as optimistic. This stage is not part of the delivered system.
- **On-device performance has not been measured yet.** This includes Jetson speed, TensorRT conversion and the full drone integration.
- **Planned next steps:** longer fine-tuning with validation-only early stopping, larger crops (320 px), wider padding (0.20–0.30), per-class Madinah error analysis, and a Jetson inference script that reads `nakhlah_model.json`.

## Team

CS499 Senior Project, College of Computer and Information Sciences, Prince Sultan University, 2026.
Supervisor: **Dr. Fatimah Alshannaq**

| Member | Role |
|---|---|
| Fatoon Aljunobi | Team Leader |
| Latifah Bin Showaish | Hardware Head |
| Rama Hassan | Programming Head |
| Shiyam Alhudaithy | Software Engineering Head |
| Lubna Alnajran | Documentation & Communications Lead |

## Acknowledgements

- Al-Madinah date palm leaf dataset: Namoun et al. (2024)
- [timm](https://github.com/huggingface/pytorch-image-models) for EdgeNeXt and the other backbones; EdgeNeXt by Maaz et al. (2022)
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO26, and [Roboflow](https://roboflow.com) for labelling and dataset hosting

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The datasets keep their own licences and are not covered by it.
