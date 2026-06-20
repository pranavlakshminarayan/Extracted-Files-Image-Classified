# Image Classification Report

This repository is a standalone duplicate of `Extracted-Files`, created for
post-processing its extracted images without altering the source dataset.

## Result

- Images processed: 1,845
- Useful hardware-design images: 129
- Non-useful images: 753
- Review required: 963
- Per-project manifests written: 144

## Folder Layout

Every image now lives under its original project folder in one of these paths:

```text
<project>/image_categories/useful/
<project>/image_categories/non_useful/
<project>/image_categories/review/
```

Each project also contains `image_classification.json`, which records the
original path, label, confidence, method, reason, and new location.

## Classification Policy

Useful images include architecture/block diagrams, schematics, timing and
waveform plots, FSM/state diagrams, pinouts, datapaths, pipelines, floorplans,
and register-oriented diagrams.

Non-useful images include logos, badges, icons, small UI fragments, screenshots,
game/video output, decorative art, and product/board photographs.

The `review` category is intentionally conservative. It contains images whose
filename or lightweight visual features were insufficient to determine whether
they are technical diagrams. It should be reviewed with a vision-capable model
or by a human before being used for model training.

## Methods Used

The GitHub-native classifier used filename/path rules, tiny-image detection,
and conservative photo-like visual heuristics. It made all moves using the
GitHub Git Data API and preserved image bytes by reusing their existing Git
blob objects.

The reproducible tool is available at `tools/classify_images.py`.
