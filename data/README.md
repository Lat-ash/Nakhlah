# Data

Images are **not stored in the git repository**. You can get them in two ways.

## 1. Download our farm photos (public)

The two farms we collected ourselves are attached to the **[`data-v1` release](https://github.com/Lat-ash/Nakhlah/releases/tag/data-v1)**:

| File | Farm | Images | With boxes | Roboflow source |
|---|---|---:|---:|---|
| `nakhlah_alammariyah_v1.zip` | Al-Ammariyah | 1,015 | 1,014 | `alamariah` v1 |
| `nakhlah_althumamah_v1.zip` | Al-Thumamah | 1,044 | 1,027 | `althumamah` v1 |

Format: YOLO (`train/images` + `train/labels` + `data.yaml`). Roboflow auto-oriented every image, removed its metadata and resized it to 640×640 (stretched); no augmentation was applied. Each box marks one leaf and is labelled with one of four classes. Photos without boxes are skipped by the notebooks.

Photos per class (by the most common box label in each photo):

| Farm | fungalSpotting | fungalWiltBlight | healthy | nutrientDeficiency | Total |
|---|---:|---:|---:|---:|---:|
| Al-Ammariyah | 385 | 300 | 140 | 189 | 1,014 |
| Al-Thumamah | 159 | 164 | 553 | 151 | 1,027 |
| **Total** | **544** | **464** | **693** | **340** | **2,041** |

Boxes are used to **crop leaves for classification**, not for object detection.

## 2. Through Roboflow (what the notebooks do)

The notebooks download the same versions from Roboflow at run time. This needs access to the team's Roboflow workspace and your own API key.

| Roboflow project | Version | Contents |
|---|---|---|
| `alamariah` | 1 | Al-Ammariyah |
| `althumamah` | 1 | Al-Thumamah |
| `combined-dataset-for-naklah` | 3 | Al-Madinah photos + the Al-Qassim mini-set (file names starting with `QD`) |

**Do not change the dataset versions.** The frozen split depends on them, and the notebooks stop with `❌ STOP: the split CHANGED` if they change.

## Al-Madinah data

The Al-Madinah images come from **Namoun et al. (2024)** and are **not redistributed here**. Get them from the original dataset and follow its licence and citation requirements. Its `insectInfestation` class is not used.

## Notes

- Leaf groups (blocks of 10 neighbouring image numbers) are a heuristic used for splitting. They are not verified physical leaves.
- If you use our farm photos, please cite this repository.
