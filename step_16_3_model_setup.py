
import torch
import numpy as np
import pandas as pd
import pydicom

from pathlib import Path
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F


# ==========================================
# STEP 16.3 - MODEL INITIALIZATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

METADATA_DIR = PROJECT_DIR / "data" / "processed" / "metadata"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 2
IMAGE_SIZE = 512

print("=" * 60)
print("STEP 16.3 - MODEL INITIALIZATION")
print("=" * 60)


# ==========================================
# 1. DEVICE CONFIGURATION
# ==========================================

device = torch.device("cpu")

print("\nDevice:", device)


# ==========================================
# 2. LOAD PRETRAINED MODEL
# ==========================================

print("\nLoading pretrained Faster R-CNN model...")

weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT

model = fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=weights
)

print("Pretrained COCO weights loaded successfully.")


# ==========================================
# 3. REPLACE CLASSIFICATION HEAD
# ==========================================

in_features = (
    model.roi_heads.box_predictor.cls_score.in_features
)

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)

model.to(device)

print("\nClassification head replaced.")
print("Number of classes:", NUM_CLASSES)
print("Classes: Background, Pneumonia Opacity")


# ==========================================
# 4. LOAD REAL VALIDATION IMAGE
# ==========================================

print("\nLoading validation image...")

val_metadata = pd.read_csv(
    METADATA_DIR / "validation_metadata.csv"
)

sample = val_metadata.iloc[0]

image_path = PROJECT_DIR / sample["image_path"]

dicom = pydicom.dcmread(str(image_path))

image = dicom.pixel_array.astype(np.float32)

if dicom.PhotometricInterpretation == "MONOCHROME1":
    image = image.max() + image.min() - image

minimum = image.min()
maximum = image.max()

if maximum > minimum:
    image = (image - minimum) / (maximum - minimum)
else:
    image = np.zeros_like(image)

image = torch.from_numpy(image).float()

image = image.unsqueeze(0).unsqueeze(0)

image = F.interpolate(
    image,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False
)

image = image.squeeze(0).repeat(3, 1, 1)

image = image.to(device)

print("Patient ID:", sample["patientId"])
print("Image shape:", tuple(image.shape))


# ==========================================
# 5. TEST MODEL FORWARD PASS
# ==========================================

print("\nTesting model inference...")

model.eval()

with torch.no_grad():

    predictions = model([image])

prediction = predictions[0]

print("Predicted boxes:", prediction["boxes"].shape)
print("Predicted labels:", prediction["labels"].shape)
print("Predicted scores:", prediction["scores"].shape)


# ==========================================
# 6. SAVE INITIALIZED MODEL
# ==========================================

checkpoint_path = (
    MODEL_DIR / "fasterrcnn_mobilenetv3_initial.pth"
)

torch.save(
    model.state_dict(),
    checkpoint_path
)

print("\nInitial model checkpoint saved:")
print(checkpoint_path)


# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("STEP 16.3 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Architecture: Faster R-CNN MobileNetV3-Large 320 FPN")
print("Pretrained weights: COCO")
print("Detection classes:", NUM_CLASSES)
print("Device: CPU")
print("Real image inference: PASSED")
print("Model checkpoint: SAVED")
