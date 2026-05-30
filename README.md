# BERR: Reliable representation and localization for real-time low-light object detection

Official PyTorch implementation of **BERR**, a reliability-guided real-time detector for low-light object detection.

BERR is designed for object detection under severe illumination degradation. Instead of reconstructing enhanced images, BERR improves detection reliability through illumination-robust representation learning, reliability-guided feature refinement, and uncertainty-aware localization.

The complete source code will be made publicly available upon acceptance of our paper. Links to the datasets used in this study are provided in the repository.

---

## Highlights

- **Reliable low-light detection** without explicit image reconstruction.
- **IIDM** extracts illumination-robust intrinsic responses before backbone feature extraction.
- **RGR** suppresses unreliable high-frequency responses during feature fusion.
- **UDR** estimates boundary ambiguity from predicted localization distributions and reweights regression supervision.
- Experiments are conducted on **ExDark**, **DARK FACE**, and **LLVIP**.

---

## Model Zoo

| Model | Dataset | Input size | mAP50 (%) | mAP75 (%) | mAP50:95 (%) | Params (M) | FPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BERR | ExDark | 640 | 83.5 | 61.9 | 55.1 | 31.4 | 52.7 |
| BERR | DARK FACE | 640 | 61.2 | - | 26.5 | 31.4 | - |
| BERR | LLVIP | 640 | 92.7 | - | 54.9 | 31.4 | - |

Notes:

- FPS is measured on a single NVIDIA RTX 4090D GPU with batch size 1.
- The reported results follow the experimental settings in the paper.
- For ExDark, mAP50, mAP75, and mAP50:95 are reported.
- For DARK FACE and LLVIP, mAP50 and mAP50:95 are reported.
- Pretrained checkpoints will be released after code cleaning.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/your-name/BERR.git
cd BERR
```

Create a conda environment:

```bash
conda create -n BERR python=3.9 -y
conda activate BERR
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python 3.9
PyTorch 2.0.1
CUDA 11.8
```

Experiments in the paper are conducted on a single NVIDIA RTX 4090D GPU.

---

## Dataset Preparation

This project uses three public low-light object detection datasets: **ExDark**, **DARK FACE**, and **LLVIP**.

The dataset links used in this study are provided here:

| Resource |                            Link                             | Extraction code |
| :---: |:-----------------------------------------------------------:| :---: |
| Datasets | [Download](https://pan.baidu.com/s/11ckriVbe8kAy6nxe2PmNvw) | `bq35` |

A recommended directory structure is:

```text
datasets/
  ExDark/
    images/
    annotations/
  DARK_FACE/
    images/
    annotations/
  LLVIP/
    images/
    annotations/
```

Please update the dataset paths in the corresponding configuration files:

```text
configs/dataset/exdark_detection.yml
configs/dataset/darkface_detection.yml
configs/dataset/llvip_detection.yml
```

For LLVIP, only visible-light images are used in this work.

---

## Training

### Train on ExDark

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/berr/berr_hgnetv2.yml
```
---

## Evaluation

Evaluate a trained checkpoint on ExDark:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/berr/berr_hgnetv2.yml \
  -r path/to/checkpoint.pth \
  --test-only
```

---

## Expected Project Structure

```text
BERR/
  configs/
    dataset/
      exdark_detection.yml
      darkface_detection.yml
      llvip_detection.yml
    BERR/
      berr_hgnetv2.yml

  datasets/
    ExDark/
    DARK_FACE/
    LLVIP/

  src/
    core/
    data/
    nn/
    solver/
    zoo/

  tools/
    train.py
    export_onnx.py
    infer.py

  requirements.txt
  README.md
```

Please make sure the dataset paths, category definitions, and annotation files are consistent with your local dataset structure.


---

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{shen2026BERR,
  title={BERR: Reliability-guided representation and regression for real-time low-light object detection},
  author={Shen, Junyi and Peng, Yong and Fu, Yu and Liu, Ming and Dong, Liquan and Kong, Lingqin},
  journal={},
  year={2026}
}
```

The citation information will be updated after publication.

---

## Acknowledgements

We thank the authors of RT-DETR for their open-source implementation.

We also thank the authors of YOLA and D-FINE.

We thank the providers of ExDark, DARK FACE, and LLVIP for making their
datasets available to the research community.
---

