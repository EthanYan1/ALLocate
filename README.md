# ALLocate

**An AI-powered self-driving microscope for low-cost acute leukemia detection**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> ALLocate (**Acute Leukemia Locate**) couples a low-cost motorized microscope stage with a **two-stage** deep learning pipeline: a **CNN region classifier** (adequate vs. blood vs. clot) and a **YOLOv8** blast / non-blast detector. This repository will hold code, notebooks, hardware notes, and release artifacts for the accompanying manuscript.

---

## At a glance

| Component | Role |
|-----------|------|
| **AAMSS** (hardware) | Automated capture of marrow smears on a conventional microscope (~\$150 parts; not including microscope) |
| **Stage 1 — Region model** | Selects diagnostically adequate tiles before cell-level analysis |
| **Stage 2 — Detection model** | Localizes nucleated cells and classifies blasts vs. non-blasts |
| **End-to-end** | Tile → regions → detections → slide-level blast fraction vs. clinical thresholds |

```mermaid
flowchart LR
  subgraph hw [Hardware]
    A[Glass slide] --> B[AAMSS scan]
    B --> C[Captured images / tiles]
  end
  subgraph ai [Two-stage AI]
    C --> D[Stage 1: Region CNN]
    D --> E[Adequate regions]
    E --> F[Stage 2: YOLOv8]
    F --> G[Blast counts & slide-level summary]
  end
```

---

## Repository layout *(skeleton — organize as code lands)*

```
ALLocate/
├── README.md                 ← you are here
├── region_classifier/      ← Stage 1: data prep, training, evaluation, inference
├── cell_detection/         ← Stage 2: data prep, training, evaluation, inference
├── pipeline/               ← End-to-end scripts: tile → region → detect → report
├── data/                   ← Data layout placeholders (no patient data in git)
├── notebooks/              ← Figure / plot reproduction for the paper
├── examples/               ← A small set of example images for documentation
├── weights/                ← Model checkpoints (release policy TBD)
```

---

## Installation

> **TODO (Harry):** Add environment name, Python version, CUDA/CPU notes, and exact `pip` / `conda` commands once dependencies are pinned.

```bash
# Placeholder — replace with real instructions
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


---

## Data & cohorts *(high level)*

Training used de-identified whole-slide and tiled data from **UCSF**; external testing included **MSKCC** digitized slides and **glass-slide** evaluation via the self-driving microscope (**SDM**). 

Purpose *(paper)* | Notes for this repo |
|-------------------|---------------------|
Region classifier train / eval | Placeholder manifests under `region_classifier/` |
Cell detector train / eval | Bounding-box format TBD in `cell_classifer/`

**TODO:** Add `data/availbility_statement.md` with directory conventions, filename patterns, and ethics / access language approved by the team.

---

## Stage 1 — Region classifier (CNN)

### Data format

On-disk layout follows **ImageNet-style** image folders (class = subdirectory name):

```
<dataset_root>/
├── train/
│   ├── adequate/
│   │   ├── <image>.png
│   │   └── ...
│   ├── blood/
│   └── clot/
└── test/
    ├── adequate/
    ├── blood/
    └── clot/
```

- Source tiles are 40× / 400×-equivalent WSI-derived patches; class names are **`adequate`**, **`blood`**, **`clot`**.
- Use **`train/`** for training and **`test/`** for held-out evaluation (add **`val/`** if you use a separate validation split).

### Training

- Framework: TensorFlow *(per manuscript)*; augmentations via Albumentations.
- Entry point: **`region_classifier/train.py`** *(stub — Ethan to implement)*.

### Evaluation

- Entry point: **`region_classifier/eval.py`** *(stub — Ethan to implement)*.


---

## Stage 2 — Cell detection (YOLOv8)

### Data format (Roboflow → YOLO)

Labels for this project were collected in **[Roboflow](https://roboflow.com/)** (private workspace / project).

- **Project type:** **Object detection** — each instance is a **bounding box** (axis-aligned rectangle).  
  Roboflow also supports **instance segmentation** (polygon / “press P” vertex tools); **that workflow is not what we used** for ALLocate’s cell detector—stick to **bounding-box** detection labels for blast vs. non-blast (+ **artifact** during training to suppress false positives).
- **Typical Roboflow flow:** create a workspace → **Upload** images → assign / open **Unannotated** images → draw boxes per class → **Generate** a version (with any augmentations in Roboflow) → **Export** in **YOLOv8** format (YOLOv5-compatible layout is fine for Ultralytics).

After export, the dataset on disk usually matches this shape (names may be `valid` vs `val` depending on export):

```
<dataset_root>/
├── data.yaml              # train/val paths, nc, names
├── train/
│   ├── images/            # .jpg / .png
│   └── labels/            # one .txt per image, YOLO format
├── valid/                 # or val/
│   ├── images/
│   └── labels/
└── test/                  # optional
    ├── images/
    └── labels/
```

**YOLO label files** (`.txt`): one line per box — `class_id x_center y_center width height` with coordinates **normalized** to `[0, 1]` relative to image width/height. **`data.yaml`** lists `train` / `val` (paths or relative dirs), `nc`, and `names` (class names in order of class id).

### Training

- **Ultralytics YOLOv8**; hyperparameter choices are summarized in the paper’s supplementary tables.
- Entry point: **`cell_detection/train.py`** *(stub — Ethan to implement; typically wraps `yolo train` or the Ultralytics API with a path to `data.yaml`)*.

### Evaluation

- Entry point: **`cell_detection/eval.py`** *(stub — Ethan to implement; validation mAP / PR, etc.)*.

### Inference *(placeholder)*

- Input: regions from Stage 1; output: boxes, class labels, optional exports for blast quantification.
- **TODO:** Batch vs. real-time inference, NMS settings, and slide-level blast aggregation in `pipeline/`.

---

## End-to-end pipeline

**TODO:** Describe how captured images or WSI tiles flow through Stage 1 → region selection (e.g. top-*k* adequate regions) → Stage 2 → **slide-level** blast percentage and comparison to clinical thresholds (e.g. ≤5% vs. ≥20%). Implementation will live under `pipeline/` with example configs.

---

## Hardware & stage control *(transparency)*

The manuscript describes a **RAMPS 1.4 + Arduino Mega 2560** setup, **NEMA 14** motors, and **3D-printed** couplings for X/Y stage actuation.

**Open item:** Final packaging for release — e.g. Arduino firmware, wiring diagram, mechanical BOM, and a short “control architecture” narrative may live under `hardware_firmware/` and `docs/`. *Decision pending.*

---

## Model weights

> **Release policy:** Whether **institutionally sensitive** fine-tuned weights can be distributed requires **Greg’s** input. Until then, this repo may ship a **public base architecture** or **sanitized checkpoint** under `weights/` with clear naming, e.g. `base/` vs. `allocate_full/` (placeholder).

**TODO:** Add checksums, license line per checkpoint, and inference-only vs. trainable flags.

---

## Examples

**TODO (Ethan):** Add **five** representative images (e.g. region classes, detection overlays, or glass-slide crops) under `examples/` with short captions in `examples/README.md`. *No PHI; de-identified or synthetic only.*

---

## Notebooks — figures & plots

**TODO (Ethan):** Curate notebooks under `notebooks/` that reproduce paper figures (ROC, PR, mAP, slide-level metrics, correlation plots, etc.), with pinned outputs or instructions to regenerate.

Suggested stubs:

- `notebooks/figures_region_classifier.ipynb`
- `notebooks/figures_cell_detection.ipynb`
- `notebooks/figures_slide_level.ipynb`

---

## Project checklist *(internal)*

- [ ] Draft README finalized *(this file — iterate)*  
- [ ] Folder structure agreed; **Ethan** reorganizes code into layout above  
- [ ] **Harry:** installation & dependency lockfiles  
- [ ] **Ethan:** five example images + notebook index for plots  
- [ ] **Greg:** decision on **official release** of sensitive model weights  
- [ ] Hardware / Arduino / control-architecture documentation approach decided  

---

## Citation

If you use this software or models, please cite the ALLocate manuscript *(details to add upon publication)*:

```bibtex
@article{allocate2025,
  title   = {An AI-Powered Self-Driving Microscope for Low-Cost Acute Leukemia Detection},
  author  = {Yan, Ethan and Sun, Shenghuan and others},
  journal = {TBD},
  year    = {2025},
  note    = {TODO: update with venue, volume, pages, DOI}
}
```

---

## License

This repository is licensed under **Creative Commons Attribution-NonCommercial 4.0 International** (**[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**). See the [`LICENSE`](LICENSE) file for the full legal text.

**Summary (not legal advice):** you may share and adapt the material **for non-commercial purposes** if you give **attribution** and point to this license. **Commercial use** is not permitted under this license without separate permission.

**Other artifacts:** model weights, datasets, or third-party dependencies may be governed by **separate** terms; see `weights/README.md` and any per-component notices.

---

## Disclaimer

This repository supports **research and education**. It is **not** a medical device and is **not** intended for primary diagnosis. Clinical use requires appropriate validation, regulatory clearance where applicable, and qualified human oversight.

---

## Contact

**TODO:** Add correspondence email, issue tracker link, and institutional affiliation lines as appropriate.
