# Model files

Model weights are **not stored in git**, because they are large binary files. Share them through Google Drive or attach them to a GitHub Release.

## Final classifier: EdgeNeXt-Base (4 classes)

Created by `notebooks/final/Nakhlah_EdgeNeXt_FINAL_Model_for_Jetson.ipynb` in `MyDrive/Nakhlah_EdgeNeXt_FINAL_model/`.

| File | What it is |
|---|---|
| `edgenext_nakhlah.pt` | PyTorch `state_dict` for `timm.create_model("edgenext_base.in21k_ft_in1k", num_classes=4)` |
| `nakhlah_model.json` | class order, `image_size` 224, `crop_padding` 0.10, `min_box_px` 20, ImageNet mean/std |
| `edgenext_nakhlah.onnx` | ONNX export (opset 17, input `image` 1×3×224×224, output `logits`) for TensorRT |
| `jetson_test_crops.zip` | 40 locked-test crops + `labels.csv`, for checking the model on the Jetson |

Class order used by the model:

```
0 fungalSpotting
1 fungalWiltBlight
2 healthy
3 nutrientDeficiency
```

Always read the class order from `nakhlah_model.json` instead of hard-coding it.

## Experimental: YOLO26s leaf-cropping model

Created by `notebooks/final/Nakhlah_YOLO26s_640_bestRunWithEdgeNet.ipynb` in `MyDrive/Nakhlah_YOLO26s_leaf_detector/`: `nakhlah_yolo26s.pt`, `nakhlah_yolo26s.onnx` and `nakhlah_yolo.json` (confidence threshold 0.10 + fallback). This model is experimental and is not part of the delivered system. See the Limitations section of the main README.
