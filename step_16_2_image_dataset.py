
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pydicom

from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ==========================================
# STEP 16.2 - PYTORCH IMAGE DATASET
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

METADATA_DIR = PROJECT_DIR / "data" / "processed" / "metadata"

IMAGE_SIZE = 512
BATCH_SIZE = 2


print("=" * 60)
print("STEP 16.2 - PYTORCH IMAGE DATASET")
print("=" * 60)


# ==========================================
# 1. CUSTOM DATASET CLASS
# ==========================================

class PneumoniaDataset(Dataset):

    def __init__(self, metadata_file, annotations_file):

        self.metadata = pd.read_csv(
            metadata_file,
            dtype={"patientId": str}
        )

        self.annotations = pd.read_csv(
            annotations_file,
            dtype={"patientId": str}
        )

        self.annotations_by_patient = {
            patient_id: group
            for patient_id, group in
            self.annotations.groupby("patientId")
        }

        self.image_size = IMAGE_SIZE

        print("\nDataset initialized:")
        print("Patients:", len(self.metadata))
        print("Annotation rows:", len(self.annotations))

    def __len__(self):
        return len(self.metadata)

    # ======================================
    # 2. LOAD AND PREPROCESS DICOM IMAGE
    # ======================================

    def load_image(self, image_path):

        dicom = pydicom.dcmread(str(image_path))

        image = dicom.pixel_array.astype(np.float32)

        if image.ndim != 2:
            raise ValueError(
                f"Expected grayscale DICOM image, got {image.shape}"
            )

        # Handle MONOCHROME1 images
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = image.max() + image.min() - image

        original_height, original_width = image.shape

        # Normalize pixel values between 0 and 1
        min_value = image.min()
        max_value = image.max()

        if max_value > min_value:
            image = (image - min_value) / (max_value - min_value)
        else:
            image = np.zeros_like(image)

        # Convert NumPy array to PyTorch tensor
        image = torch.from_numpy(image).float()

        # Add batch and channel dimensions
        image = image.unsqueeze(0).unsqueeze(0)

        # Resize image
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )

        # Remove batch dimension
        image = image.squeeze(0)

        # Convert grayscale into three channels
        image = image.repeat(3, 1, 1)

        return image, original_height, original_width

    # ======================================
    # 3. LOAD AND SCALE BOUNDING BOXES
    # ======================================

    def load_boxes(
        self,
        patient_id,
        original_height,
        original_width
    ):

        patient_annotations = self.annotations_by_patient.get(
            patient_id
        )

        boxes = []

        if patient_annotations is not None:

            valid_annotations = patient_annotations.dropna(
                subset=["x", "y", "width", "height"]
            )

            valid_annotations = valid_annotations[
                valid_annotations["Target"] == 1
            ]

            scale_x = self.image_size / original_width
            scale_y = self.image_size / original_height

            for row in valid_annotations.itertuples():

                x1 = max(0, float(row.x) * scale_x)
                y1 = max(0, float(row.y) * scale_y)

                x2 = min(
                    self.image_size,
                    (float(row.x) + float(row.width)) * scale_x
                )

                y2 = min(
                    self.image_size,
                    (float(row.y) + float(row.height)) * scale_y
                )

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        return boxes

    # ======================================
    # 4. GET ONE DATASET SAMPLE
    # ======================================

    def __getitem__(self, index):

        record = self.metadata.iloc[index]

        patient_id = record["patientId"]

        image_path = PROJECT_DIR / record["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        image, original_height, original_width = (
            self.load_image(image_path)
        )

        boxes = self.load_boxes(
            patient_id,
            original_height,
            original_width
        )

        # Object detection class:
        # 1 = Pneumonia opacity

        labels = torch.ones(
            (len(boxes),),
            dtype=torch.int64
        )

        area = (
            (boxes[:, 2] - boxes[:, 0]) *
            (boxes[:, 3] - boxes[:, 1])
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros(
                (len(boxes),),
                dtype=torch.int64
            )
        }

        return image, target


# ==========================================
# 5. CREATE TRAINING DATASET
# ==========================================

train_dataset = PneumoniaDataset(
    METADATA_DIR / "train_metadata.csv",
    METADATA_DIR / "train_annotations.csv"
)


# ==========================================
# 6. CREATE VALIDATION DATASET
# ==========================================

val_dataset = PneumoniaDataset(
    METADATA_DIR / "validation_metadata.csv",
    METADATA_DIR / "validation_annotations.csv"
)


# ==========================================
# 7. CUSTOM BATCH COLLATE FUNCTION
# ==========================================

def detection_collate_fn(batch):

    return tuple(zip(*batch))


# ==========================================
# 8. CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=detection_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=detection_collate_fn
)


# ==========================================
# 9. TEST TRAINING DATASET
# ==========================================

print("\nTesting training DataLoader...")

images, targets = next(iter(train_loader))

print("\nTraining batch results:")

for index, (image, target) in enumerate(
    zip(images, targets)
):

    print(f"\nImage {index + 1}")
    print("Image shape:", tuple(image.shape))
    print("Image dtype:", image.dtype)
    print("Pixel minimum:", image.min().item())
    print("Pixel maximum:", image.max().item())
    print("Bounding boxes:", len(target["boxes"]))
    print("Labels:", target["labels"].tolist())

    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.isfinite(image).all()
    assert torch.isfinite(target["boxes"]).all()

    assert (
        (target["boxes"] >= 0).all()
        and (target["boxes"] <= IMAGE_SIZE).all()
    )


# ==========================================
# 10. TEST VALIDATION DATASET
# ==========================================

print("\nTesting validation dataset...")

val_image, val_target = val_dataset[0]

print("Validation image shape:", tuple(val_image.shape))
print("Validation bounding boxes:", len(val_target["boxes"]))


# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("STEP 16.2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Training patients:", len(train_dataset))
print("Validation patients:", len(val_dataset))

print("Image size:", IMAGE_SIZE, "x", IMAGE_SIZE)
print("Image channels: 3")
print("Batch size:", BATCH_SIZE)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

print("\nImage preprocessing: PASSED")
print("Bounding-box processing: PASSED")
print("Training DataLoader: PASSED")
print("Validation Dataset: PASSED")