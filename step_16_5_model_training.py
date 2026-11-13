
import torch
import numpy as np
import time
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader, Subset

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from step_16_2_image_dataset import (
    train_dataset,
    detection_collate_fn
)


# ==========================================
# STEP 16.5 - MODEL TRAINING
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

INITIAL_MODEL = MODEL_DIR / "fasterrcnn_mobilenetv3_initial.pth"

DEVICE = torch.device("cpu")

SAMPLES_PER_CLASS = 1000
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 0.001

MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)

print("=" * 60)
print("STEP 16.5 - PNEUMONIA MODEL TRAINING")
print("=" * 60)


# ==========================================
# 1. PREPARE BALANCED TRAINING SUBSET
# ==========================================

metadata = train_dataset.metadata

positive_indices = metadata.index[
    metadata["Target"] == 1
].tolist()

negative_indices = metadata.index[
    metadata["Target"] == 0
].tolist()

if len(positive_indices) < SAMPLES_PER_CLASS:
    raise ValueError("Insufficient positive training samples.")

if len(negative_indices) < SAMPLES_PER_CLASS:
    raise ValueError("Insufficient negative training samples.")

rng = np.random.default_rng(42)

selected_positive = rng.choice(
    positive_indices,
    size=SAMPLES_PER_CLASS,
    replace=False
).tolist()

selected_negative = rng.choice(
    negative_indices,
    size=SAMPLES_PER_CLASS,
    replace=False
).tolist()

selected_indices = selected_positive + selected_negative

rng.shuffle(selected_indices)

training_subset = Subset(
    train_dataset,
    selected_indices
)

train_loader = DataLoader(
    training_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=detection_collate_fn
)

print("\nTraining dataset prepared.")
print("Positive images:", len(selected_positive))
print("Negative images:", len(selected_negative))
print("Total images:", len(training_subset))
print("Training batches:", len(train_loader))


# ==========================================
# 2. INITIALIZE MODEL
# ==========================================

print("\nLoading model architecture...")

model = fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=None,
    weights_backbone=None
)

in_features = (
    model.roi_heads.box_predictor.cls_score.in_features
)

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    2
)

checkpoint = torch.load(
    INITIAL_MODEL,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)

print("Initial pretrained checkpoint loaded.")


# ==========================================
# 3. FREEZE BACKBONE
# ==========================================

for parameter in model.backbone.parameters():
    parameter.requires_grad = False

# RPN and ROI heads remain trainable

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

print("\nBackbone: Frozen")
print("RPN: Trainable")
print("ROI detection head: Trainable")

print(
    "Trainable parameters:",
    sum(p.numel() for p in trainable_parameters)
)


# ==========================================
# 4. OPTIMIZER
# ==========================================

optimizer = torch.optim.SGD(
    trainable_parameters,
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005
)


# ==========================================
# 5. TRAINING
# ==========================================

epoch_history = []

overall_start = time.perf_counter()

for epoch in range(EPOCHS):

    model.train()

    # Keep frozen backbone in evaluation mode
    model.backbone.eval()

    running_loss = 0.0
    epoch_start = time.perf_counter()

    print("\n" + "-" * 60)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("-" * 60)

    for batch_index, (images, targets) in enumerate(train_loader):

        images = [
            image.to(DEVICE)
            for image in images
        ]

        targets = [
            {
                key: value.to(DEVICE)
                for key, value in target.items()
            }
            for target in targets
        ]

        optimizer.zero_grad()

        loss_dict = model(images, targets)

        total_loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(total_loss):
            raise ValueError(
                f"Invalid loss at batch {batch_index + 1}"
            )

        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item()

        if (
            (batch_index + 1) % 50 == 0
            or batch_index + 1 == len(train_loader)
        ):

            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss: {total_loss.item():.4f}"
            )

    average_loss = running_loss / len(train_loader)

    epoch_time = time.perf_counter() - epoch_start

    epoch_history.append({
        "epoch": epoch + 1,
        "average_loss": average_loss,
        "training_time_seconds": epoch_time
    })

    print("\nEpoch completed.")
    print("Average loss:", round(average_loss, 4))
    print("Epoch time:", round(epoch_time, 2), "seconds")

    # Save checkpoint after every epoch
    epoch_checkpoint = (
        MODEL_DIR /
        f"fasterrcnn_mobilenetv3_epoch_{epoch + 1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "average_loss": average_loss,
            "image_size": 512,
            "training_samples": len(training_subset)
        },
        epoch_checkpoint
    )

    print("Checkpoint saved:", epoch_checkpoint.name)


# ==========================================
# 6. SAVE FINAL TRAINED MODEL
# ==========================================

final_checkpoint = (
    MODEL_DIR / "fasterrcnn_mobilenetv3_trained.pth"
)

torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_history": epoch_history,
        "image_size": 512,
        "training_samples": len(training_subset)
    },
    final_checkpoint
)

overall_time = time.perf_counter() - overall_start


# ==========================================
# 7. SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(epoch_history)

history_df.to_csv(
    MODEL_DIR / "training_history.csv",
    index=False
)

print("\n" + "=" * 60)
print("STEP 16.5 - TRAINING COMPLETED")
print("=" * 60)

print("\nTraining history:")
print(history_df.to_string(index=False))

print("\nTotal training time:", round(overall_time, 2), "seconds")

print("\nFinal model:", final_checkpoint)
print("Training history: models/training_history.csv")

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
