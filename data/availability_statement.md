# Data availability statement

**Manuscript status.** The ALLocate team is **actively working on the accompanying paper**. This statement may be updated as the manuscript is revised and as institutional approvals evolve.

## What this repository contains

This repository hosts **software, documentation, and (where released) model-related artifacts**. It does **not** include raw patient imaging, whole-slide image banks, or other materials that could enable re-identification. **Do not commit** identifiable patient data or institution-internal paths.

## Cohorts and use *(high level)*

Training and development used **de-identified** whole-slide and tiled data from **UCSF**. External evaluation included **MSKCC** digitized slides and **glass-slide** imaging acquired with the self-driving microscope (**SDM**) / **AAMSS** hardware setup described in the manuscript.

Specific numbers, inclusion criteria, and split definitions will appear in the paper and any supplementary materials.

## Sharing, weights, and future releases

- **Research data.** We are in discussion with our academic institutions about sharing de-identified imaging data and related materials. **Timing and scope** of any public release are expected to align with **publication and institutional approvals**, subject to ethics, privacy, and data-use agreements. Until those terms are settled, **no patient-level or restricted datasets are distributed through this repository**.
- **Model checkpoints.** The paper is **in revision**; checkpoint release is **not final**. See the dedicated [**model statement**](../weights/model_statement.md) (*please wait*).

For the most detailed, publication-aligned wording, see the **Data availability** (and related) sections of the manuscript once it is available.

## Examples folder

Illustrative images under `examples/` (when present) are for **documentation only**. They are not a dataset release. See [`examples/README.md`](../examples/README.md) and this file for constraints on what may be shown.

## Contact

For collaboration or data questions after the paper is public, use the contact information in the repository README or the published article.
