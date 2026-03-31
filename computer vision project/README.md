# 🌿 Plant Disease Detection using Computer Vision

A deep learning web application that detects diseases in plant leaves from a photo. Built as part of a Computer Vision course using PyTorch and Gradio.

---

## 🔍 Problem Statement

Plant diseases cause significant crop losses every year, especially in smallholder farms where access to agricultural experts is limited. Early detection is critical but often requires trained agronomists. This project builds an accessible tool that allows anyone with a smartphone to photograph a leaf and get an instant disease diagnosis.

---

## 🧠 How It Works

1. A photo of a plant leaf is uploaded via the web interface.
2. The image is preprocessed (resized to 224×224, normalized).
3. A fine-tuned **ResNet50** model classifies it into one of **38 categories** (healthy or diseased) covering crops like tomato, potato, apple, corn, grape, and more.
4. The top-5 predictions with confidence scores are displayed.

**Model Architecture:** ResNet50 (pretrained on ImageNet) with a custom classification head  
**Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — ~54,000 leaf images  
**Interface:** Gradio web app (runs locally or can be shared publicly)

---

## 📁 Project Structure

```
plant-disease-detection/
├── app.py               # Main application (model + Gradio UI + training script)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── plant_disease_model.pth  # (after training) Saved model weights
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

### Option A: Use pretrained backbone (no training needed)
```bash
python app.py
```
> ⚠️ Without fine-tuned weights, predictions will not be accurate. See Option B to train the model.

### Option B: Train on PlantVillage dataset first

1. Download the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Organize it into `data/PlantVillage/train/` and `data/PlantVillage/val/` subfolders (one folder per class)
3. Run training from Python:

```python
from app import train
train("data/PlantVillage", epochs=10)
```

4. After training completes, the best weights are saved to `plant_disease_model.pth`
5. Run the app:
```bash
python app.py
```

### Accessing the app
After running, open the URL shown in the terminal (e.g., `http://127.0.0.1:7860`).  
A public shareable link is also generated automatically.

---

## 🌱 Supported Plants & Diseases

| Plant | Conditions Detected |
|-------|-------------------|
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Corn | Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy |
| Grape | Black rot, Esca, Leaf blight, Healthy |
| + more | Pepper, Peach, Strawberry, Cherry, Blueberry, Orange, Soybean, Squash, Raspberry |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | ResNet50 |
| Input size | 224 × 224 |
| Classes | 38 |
| Reported accuracy (PlantVillage) | ~95%+ (literature) |

---

## 🛠️ Key Technical Concepts Used

- **Transfer Learning** — pretrained ResNet50 backbone, custom classification head
- **Data Augmentation** — random crops, flips, color jitter during training
- **Image Preprocessing** — resize, normalize with ImageNet mean/std
- **Softmax classification** — multi-class probability output
- **Gradio** — rapid deployment of ML models as web apps

---

## 🤝 Acknowledgements

- Dataset: [PlantVillage](https://plantvillage.psu.edu/) by Penn State University
- Model: ResNet50 via [torchvision](https://pytorch.org/vision/stable/models.html)
- Interface: [Gradio](https://gradio.app/)
