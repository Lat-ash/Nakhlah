# Data

Images are **not stored in this repository**. The notebooks download them from Roboflow at run time.

| Roboflow project | Version | Contents |
|---|---|---|
| `alamariah` | 1 | Al-Ammariyah farm, 1,014 photos with boxes (4 classes) |
| `althumamah` | 1 | Al-Thumamah farm, 1,027 photos with boxes (4 classes) |
| `combined-dataset-for-naklah` | 3 | Al-Madinah photos (Namoun et al., 2024) and the Al-Qassim mini-set; file names starting with `QD` are Qassim |

Classes: `fungalSpotting`, `fungalWiltBlight`, `healthy`, `nutrientDeficiency`. The Madinah `insectInfestation` boxes are skipped.

**Do not change the dataset versions.** The frozen split depends on them. If a version changes, the notebooks stop with `❌ STOP: the split CHANGED`.

To get access, ask the team for an invitation to the Roboflow workspace. Each person should use their own API key.

The Al-Madinah images come from Namoun et al. (2024) and keep that dataset's licence and citation requirements.
