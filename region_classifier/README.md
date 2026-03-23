# Stage 1 — Region classifier *(skeleton)*

**Planned contents**

- `configs/` — model architecture, hyperparameters, augmentations  
- `data/` — symlinks or pointers to `../data/` (no raw PHI in git)  
- `train.py` / `eval.py` — entry points *(stubs; Ethan)*; `inference` TBD in `pipeline/`  
- `metrics/` — ROC, per-class AUC, confusion matrices  

Training data: **adequate / blood / clot** region tiles (UCSF TR/ER in paper).
