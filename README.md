# Semantic Segmentation Using U-Net

This repository provides an implementation of **semantic segmentation** using **U-Net architecture** on the **Cityscapes dataset**. The model is trained to classify different regions of an image, such as roads, buildings, and vegetation.

## 📌 Project Overview

- **Dataset**: Uses Cityscapes images and segmentation masks.
- **Model**: Implements U-Net for pixel-wise classification.
- **Training**: Uses **CrossEntropyLoss** and **Adam optimizer**.
- **Evaluation**: Computes **Mean IoU (Intersection over Union)**.
- **Uncertainty Estimation**: Visualizes entropy-based uncertainty heatmaps.

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/muditbaid/Active_Semantic_Segmentation.git
cd Active_Semantic_Segmentation
```

### 2️⃣ Install Dependencies

Create a virtual environment and install required libraries:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

### 3️⃣ Run the Pipeline

To train, evaluate, and visualize results, execute the main scripts:

#### **Train the U-Net Model**

```bash
python train.py
```

#### **Evaluate the Model Performance**

```bash
python evaluate.py
```
## 📥 Download Pretrained Model
The trained U-Net model is available for download:

🔗 [Download U-Net.pth from Google Drive](https://drive.google.com/file/d/1vypggddm-axJzBrzWr71ICbsfkMPPKPf/view?usp=sharing)

After downloading, place `U-Net.pth` in the `models/` directory before running `evaluate.py`.

## 📂 Repository Structure

```
.
├── models/               # Stores trained U-Net model weights
├── src/                  # Core scripts
│   ├── dataset_loader.py # Loads and preprocesses dataset
│   ├── unet_model.py     # Implements U-Net architecture
│   ├── train.py          # Trains the model
│   ├── evaluate.py       # Evaluates model performance
├── results/              # Stores segmentation outputs & metrics
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
```

## 🛠️ Future Improvements

- Experiment with **ResNet-based encoders** for better feature extraction.
- Implement **Self-Attention Mechanisms** for improved segmentation.
- Extend to **medical imaging datasets** (e.g., Brain Tumor Segmentation).

## 📌 References
- [ViewAL](https://github.com/nihalsid/ViewAL)
- [DIAL](https://github.com/alteia-ai/DIAL)
- [DISCA](https://github.com/delair-ai/DISCA)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [PyTorch Documentation](https://pytorch.org/)


## 📝 License

This project is for research and educational purposes. Free to use with attribution.

---

Feel free to contribute by submitting a pull request! 🚀
