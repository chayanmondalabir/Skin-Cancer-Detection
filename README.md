# Skin Cancer Detection using a Modified AlexNet Architecture

A deep learning system for automated melanoma detection from dermoscopic skin lesion images. This repository implements a **Modified AlexNet** architecture — replacing the classifier's border layers with custom Dense layers — combined with data augmentation and K-Fold cross-validation to classify skin lesions as **Melanoma** or **Not Melanoma**. The proposed model achieves **100% classification accuracy**, outperforming existing CNN and classical ML-based approaches.

> Official implementation accompanying the paper: *"Enhanced AlexNet Model for Improved Skin Cancer Detection: A Novel Approach"*, published at the **International Conference on Paradigms of Communication, Computing and Data Analytics (PCCDA)**, Springer, 2024.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Working Procedure](#working-procedure)
  - [Data Augmentation](#data-augmentation)
  - [Proposed CNN Architecture](#proposed-cnn-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Overview

Melanoma is the deadliest form of skin cancer, and its visual similarity to benign nevi makes accurate differentiation challenging even for experienced dermatologists. Early detection dramatically improves survival rates, yet conventional diagnostic workflows remain time-consuming and resource-intensive.

This project proposes an intelligent, deep-learning-based diagnostic system built on a **modified AlexNet backbone**. The convolutional feature extractor of AlexNet is retained, while the original border/classifier layers are replaced with a custom stack of Dense, Batch Normalization, and Dropout layers, followed by a sigmoid output for binary classification. The model is trained and validated using **10-fold cross-validation** on an augmented dermoscopic image dataset, achieving perfect classification accuracy on the held-out test data.

## Key Features

- 🔬 Modified AlexNet architecture with custom Dense classifier head for binary skin lesion classification
- 🖼️ Data augmentation pipeline (rotation, width/height shift, shear, zoom, horizontal flip) to address limited dataset size
- 🔁 Robust 10-fold cross-validation for reliable performance evaluation
- 📊 Comprehensive evaluation using precision, recall, F1-score, and confusion matrices
- 🏆 Achieves **100% classification accuracy**, outperforming CNN, HOG+SVM, LDA+CNN, and FCNN+GoogleNet baselines
- ⚙️ Trained with Cross-Entropy Loss and SGD optimizer using PyTorch

## Repository Structure

```
Skin-Cancer-Detection/
├── code/                  # Source code for the proposed Modified AlexNet model
├── proposed diagram/      # Architecture diagram of the proposed model
├── results/               # Result visualizations (dataset samples, confusion matrix, training curves, comparison)
├── .gitignore             # Ignored files and directories
├── LICENSE                # License information
└── README.md              # Project documentation (this file)
```

## Dataset

The dataset was sourced from **Kaggle**, supplemented with additional melanoma images collected from **Google Images** and **GitHub** to address class imbalance.

| Class | Total Images |
|---|---|
| Melanoma | 119 |
| Not Melanoma | 87 |


After applying data augmentation (rotation, width/height shift, shear, zoom, horizontal flip), the dataset was expanded to **500 images per class** (1,000 images total).

## Methodology

### Working Procedure

1. Images are resized to **227 × 227** pixels and converted into normalized PyTorch tensors.
2. Data augmentation is applied to compensate for the limited dataset size.
3. The dataset is partitioned using **K-Fold cross-validation (k = 10)**.
4. For each fold, the model is trained and validated over multiple epochs, with training/validation loss and accuracy tracked throughout.
5. **Cross-Entropy Loss** is used with **Stochastic Gradient Descent (SGD)** as the optimizer, using tuned learning rate, momentum, and weight decay.
6. The final trained model is saved for inference and future use.

### Data Augmentation

To mitigate the risk of overfitting on a small dataset, the following augmentation techniques were applied:

- Rotation
- Width shift
- Height shift
- Shear
- Zoom
- Horizontal flip

This expanded the dataset from 206 original images to **1,000 augmented images** (500 Melanoma + 500 Not Melanoma).

### Proposed CNN Architecture

The proposed network builds on the pretrained **AlexNet** convolutional backbone, with the original classifier layers replaced by a custom head:

- **AlexNet convolutional layers** (retained, transfer-learned feature extractor)
- **Global Average Pooling 2D**
- **Dense (512 units)** → Batch Normalization → ReLU → Dropout (0.3)
- **Dense (256 units)** → Batch Normalization → ReLU → Dropout (0.3)
- **Dense (1 unit)** → Sigmoid activation (binary classification: Melanoma vs. Not Melanoma)

Three convolutional blocks (with decreasing channel depth: 512 → 256) are each followed by batch normalization, ReLU activation, and max-pooling, before the final sigmoid classification layer.

**Training configuration:** 30 epochs, **SGD optimizer**, learning rate = 0.001, drop factor = 0.2, 10-fold cross-validation.

## Results

### Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Melanoma | 1.00 | 1.00 | 1.00 | 69 |
| Not Melanoma | 1.00 | 1.00 | 1.00 | 55 |

### Cross-Validation Performance Summary

| Fold | Training Accuracy | Test Accuracy |
|---|---|---|
| 1 | 71.29% | 63.71% |
| 2 | 75.78% | 74.67% |
| 3 | 89.99% | 74.67% |
| 4 | 93.72% | 80.00% |
| 5 | 100.00% | 98.65% |
| 6–10 | 100.00% | 100.00% |

### Comparison with Existing Literature

| Reference | Year | Method | Accuracy |
|---|---|---|---|
| Hossin et al. | 2020 | CNN | 93.58% |
| Daghrir et al. | 2020 | CNN | 85.5% |
| Babu & Peter | 2021 | HOG + SVM features | 76% |
| Namozov & Cho | 2018 | LDA + CNN | 85.4% |
| Khan et al. | 2019 | FCNN + GoogleNet | 88.2% |
| **Proposed** | **2024** | **Modified AlexNet** | **100%** |


**Key Takeaway:** The proposed Modified AlexNet architecture consistently outperforms existing CNN-based and classical machine learning approaches for melanoma detection, achieving perfect precision, recall, and F1-score on the held-out test data from fold 6 onward.

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Skin-Cancer_Detection.git
cd Skin-Cancer_Detection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:** `torch`, `torchvision`, `scikit-learn`, `numpy`, `pandas`, `opencv-python`, `matplotlib`, `seaborn`

## Usage

```bash
# Train and evaluate the proposed Modified AlexNet model with 10-fold cross-validation
python code/train.py --epochs 30 --optimizer sgd --lr 0.001 --k_folds 10
```

> Update dataset paths and hyperparameters in the corresponding config file before running.

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{biswas2024enhanced,
  title={Enhanced AlexNet Model for Improved Skin Cancer Detection: A Novel Approach},
  author={Biswas, Subir and Islam, Md Ariful and Abir, Chayan Mondal and Muduli, Debendra},
  booktitle={International Conference on Paradigms of Communication, Computing and Data Analytics},
  pages={297--308},
  year={2024},
  organization={Springer}
}
```

📄 **Full paper:** [SpringerLink – doi.org/10.1007/978-981-97-7946-8_22](https://link.springer.com/chapter/10.1007/978-981-97-7946-8_22)

## License

This project is licensed under the terms specified in the [LICENSE](./LICENSE) file.

## Contact

**Chayan Mondal Abir**
📧 abirchayan2000@gmail.com
**Subir Biswas**
📧 subirbiswas192001@gmail.com

For questions, issues, or collaboration inquiries, please open a GitHub issue or reach out via email.

## Acknowledgments

This work was conducted at C.V. Raman Global University, Bhubaneswar, Odisha, India, as part of ongoing research in deep-learning-based medical image analysis.
