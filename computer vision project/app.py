"""
Plant Disease Detection using Computer Vision
=============================================
Uses a pretrained ResNet50 model fine-tuned on the PlantVillage dataset
to classify plant leaf images as healthy or diseased.

Dependencies:
    pip install torch torchvision pillow gradio
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

# ─────────────────────────────────────────────
# 1. CLASS LABELS
#    Subset of PlantVillage dataset (38 classes)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper___Bacterial_spot",
    "Pepper___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────
# 2. MODEL SETUP
#    ResNet50 with custom final layer for 38 classes
# ─────────────────────────────────────────────
def build_model(weights_path: str = None) -> nn.Module:
    """
    Builds a ResNet50 model modified for plant disease classification.

    Args:
        weights_path: Optional path to saved model weights (.pth file).
                      If None, returns the base model (random final layer).

    Returns:
        torch.nn.Module ready for inference.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"[INFO] Loaded weights from {weights_path}")
    else:
        print("[INFO] No fine-tuned weights found — using ImageNet pretrained backbone.")
        print("       Predictions will not be accurate until you train/download weights.")

    model.eval()
    return model


# ─────────────────────────────────────────────
# 3. IMAGE PREPROCESSING
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
])


# ─────────────────────────────────────────────
# 4. INFERENCE
# ─────────────────────────────────────────────
def predict(image: Image.Image, model: nn.Module) -> dict:
    """
    Runs inference on a single PIL image.

    Args:
        image: PIL Image (RGB).
        model: Loaded PyTorch model.

    Returns:
        Dict mapping class label → confidence score (top 5).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = TRANSFORM(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    # Get top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, k=5)

    results = {}
    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        label = CLASS_NAMES[idx].replace("___", " — ").replace("_", " ")
        results[label] = round(prob, 4)

    return results


# ─────────────────────────────────────────────
# 5. GRADIO WEB INTERFACE
# ─────────────────────────────────────────────
def create_demo(model: nn.Module) -> gr.Interface:
    """
    Wraps the model in a Gradio web demo.
    """
    def gradio_predict(img):
        if img is None:
            return {"Error": 1.0}
        pil_img = Image.fromarray(img)
        return predict(pil_img, model)

    demo = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Image(label="Upload a leaf image"),
        outputs=gr.Label(num_top_classes=5, label="Disease Prediction"),
        title="🌿 Plant Disease Detector",
        description=(
            "Upload a clear photo of a plant leaf to detect potential diseases.\n"
            "Supports 38 categories across tomato, potato, apple, corn, and more.\n"
            "Dataset: PlantVillage | Model: ResNet50"
        ),
        examples=[],  # Add example image paths here if available
        theme=gr.themes.Soft(),
    )
    return demo


# ─────────────────────────────────────────────
# 6. TRAINING SCRIPT (Optional — run separately)
# ─────────────────────────────────────────────
def train(data_dir: str, epochs: int = 10, save_path: str = "plant_disease_model.pth"):
    """
    Fine-tunes the model on the PlantVillage dataset.

    Args:
        data_dir: Root directory with 'train/' and 'val/' subfolders.
        epochs: Number of training epochs.
        save_path: Where to save the trained weights.

    Usage:
        from app import train
        train("data/PlantVillage", epochs=10)
    """
    from torchvision import datasets
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = TRANSFORM

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"),   val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

    model = build_model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Training phase ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total

        # ── Validation phase ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")

    print(f"\n[DONE] Best validation accuracy: {best_val_acc:.2f}%")
    print(f"[DONE] Model saved to: {save_path}")


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model(weights_path="plant_disease_model.pth")
    demo = create_demo(model)
    demo.launch(share=True)
