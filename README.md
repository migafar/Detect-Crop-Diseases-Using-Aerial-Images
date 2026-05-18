# Crop Disease Detection from Aerial Images

Farmers lose crops. A lot. By the time a disease is visible at eye level, it has usually already spread. This project tries to catch those problems earlier, from above — using drone imagery and a fine-tuned EfficientNet-B3 to classify crop diseases before they become field-wide disasters.

---

## What This Actually Does

Given an aerial image of a crop field, the model predicts which disease class it belongs to. The full pipeline covers:

- Downloading and exploring the dataset (class balance checks, image size distributions, sample previews)
- Stratified train/validation/test split — 70 / 15 / 15
- Albumentations-based augmentation tailored to drone imagery (rotation, brightness shifts, Gaussian noise for sensor simulation)
- Weighted sampling to deal with class imbalance without discarding data
- Transfer learning on EfficientNet-B3 pre-trained on ImageNet
- Training with AdamW + CosineAnnealing LR scheduler
- Per-class recall reporting and confusion matrix visualization

---

## Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, EfficientNet (efficientnet_pytorch) |
| Augmentation | Albumentations |
| Data | Pandas, KaggleHub |
| Evaluation | Scikit-learn (accuracy, recall, classification report, confusion matrix) |
| Visualization | Matplotlib, Seaborn |
| Hardware | CUDA (tested on Colab T4 GPU) |

---

## Dataset

Sourced from Kaggle: [`killa92/crop-disease-image-classification-dataset`](https://www.kaggle.com/datasets/killa92/crop-disease-image-classification-dataset)

The dataset ships with a `meta_data.csv` linking image IDs to class labels, and a `class_names.json` mapping indices to disease names. The class distribution is not balanced — something the training pipeline explicitly handles.

---

## Augmentation Strategy

Drone footage comes from all angles, altitudes, and lighting conditions. The augmentation pipeline tries to reflect that reality rather than just padding the dataset:

```python
A.HorizontalFlip(p=0.5)          # drone flies in any direction
A.VerticalFlip(p=0.3)
A.Rotate(limit=25, p=0.5)
A.RandomBrightnessContrast(...)  # sun angle changes
A.HueSaturationValue(...)        # color variation across growth stages
A.GaussNoise(...)                # simulates camera sensor noise
A.GaussianBlur(blur_limit=(3,5)) # motion blur at higher speeds
```

Validation and test sets get no augmentation — just resize and normalize. That's intentional: you want to see how the model performs on real, unmodified images.

---

## Model

EfficientNet-B3 with its final fully connected layer swapped out to match the number of disease classes:

```python
model = EfficientNet.from_pretrained("efficientnet-b3")
model._fc = nn.Linear(in_features, num_classes)
```

All other weights start from ImageNet pretraining. ~12M parameters, all trainable.

**Optimizer:** Adam (`lr=3e-4`, `weight_decay=1e-5`)  
**Scheduler:** CosineAnnealingLR (`eta_min=1e-6`)  
**Loss:** CrossEntropyLoss  
**Early stopping:** patience = 5 epochs (best model saved by validation accuracy)

---

## Class Imbalance

The dataset has an imbalance problem. Rather than oversampling with synthetic images or dropping the majority class, the training loader uses `WeightedRandomSampler` — underrepresented classes get sampled more often within each batch:

```python
class_weights  = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_df['label']]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
```

---

## Evaluation

Beyond overall accuracy, the notebook reports **per-class recall** — which matters more here. Misclassifying a diseased crop as healthy is far worse than the reverse, so recall is the metric to watch.

Results also include:
- Full `classification_report` (precision, recall, F1 per class)
- Raw and normalized confusion matrices

---

## How to Run

**1. Clone / open in Colab**

The notebook is Colab-ready. Set runtime to GPU (T4 recommended).

**2. Install dependencies**

```bash
pip install torch torchvision albumentations timm kaggle efficientnet_pytorch scikit-learn matplotlib seaborn pandas
```

**3. Dataset**

```python
import kagglehub
path = kagglehub.dataset_download("killa92/crop-disease-image-classification-dataset")
```

Make sure your Kaggle API token is configured (`~/.kaggle/kaggle.json`).

**4. Run cells in order**

No configuration needed beyond that. Seeds are fixed at 42 throughout.

---



## Notes

- The notebook was built and tested on Google Colab with a T4 GPU. Local runs will work but training will be slow on CPU.
- Only 2 epochs were used in the current run. Increasing to 15–30 with early stopping enabled is recommended for production use.
- If image files are missing from the downloaded dataset, the notebook filters them out automatically before creating DataLoaders.
