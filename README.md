# Weakly Supervised Pneumonia Localization (CNN + CAM)

Re-implementation of *Weakly Supervised Pneumonia Localization* (Huang, Cortes, Monam; Stanford).
A 10-conv-layer CNN classifies chest X-rays (pneumonia / normal); a Class Activation Map (CAM)
then localises the opacity **without any bounding-box labels**. LR/SVM/RF are classification baselines.

## Pipeline
1. `prepare_data.py` – read RSNA DICOMs, keep *Lung Opacity* + *Normal*, resize 1024→128, cache.
2. 70/20/10 stratified train/val/test split; per-pixel mean/std normalisation.
3. 2-channel input: original + lung-segmented image.
4. `train.py` – CNN (3x3 convs, ReLU, GAP, one FC), Adam, lr 1e-4, 20 epochs.
5. `cam.py` – CAM → threshold → DFS clustering → bounding boxes (2-std filter) → IoU.

## Setup
```bash
git clone <your-repo> && cd pneumonia-cam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Download the data from Kaggle "RSNA Pneumonia Detection Challenge" (stage_2 files) and unzip to `data/`,
so that it contains `stage_2_train_images/`, `stage_2_train_labels.csv`, `stage_2_detailed_class_info.csv`.

## Run
```bash
cd src
python prepare_data.py --data_dir ../data --out cache.npz   # once
python baselines.py                                         # LR / SVM / RF accuracies
python train.py --epochs 20                                 # saves best.pt
python evaluate.py                                          # accuracy, confusion matrix, IoU
python demo.py --image ../data/stage_2_train_images/<id>.dcm --out result.png
```

## Notes / deviations from the paper
- BatchNorm added after each conv for stable training.
- Lung segmentation uses an Otsu heuristic, not a trained U-Net (see Extensions).
- IoU = overlap of union of predicted boxes vs union of GT boxes, per true-positive image.
- Expected (paper) numbers: test acc ≈ 92%, CAM IoU ≈ 0.15, R-CNN ≈ 0.13.

## Extensions (for extra credit / discussion)
- Train a U-Net lung segmenter (Montgomery / Shenzhen lung-mask datasets).
- Tune CAM threshold on the validation set (try 0.2–0.7).
- Use 256x256 inputs; try transfer learning (ImageNet / CheXNet weights).
