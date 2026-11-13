
import time
import numpy as np
import pandas as pd
import torch

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
# STEP 16.7 - RESUME MODEL TRAINING
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

CHECKPOINT_PATH = (
    MODEL_DIR / "fasterrcnn_mobilenetv3_epoch_1.pth"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "fasterrcnn_mobilenetv3_trained.pth"
)

DEVICE = torch.device("cpu")

SAMPLES_PER_CLASS = 1000
BATCH_SIZE = 2
TARGET_EPOCHS = 3

LEARNING_RATE = 0.001

torch.set_num_threads(4)


print("=" * 60)
print("STEP 16.7 - RESUME MODEL TRAINING")
print("=" * 60)


# ==========================================
# 1. VERIFY CHECKPOINT
# ==========================================

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

starting_epoch = checkpoint["epoch"]

if starting_epoch >= TARGET_EPOCHS:

    raise ValueError(
        "Checkpoint has already completed the target epochs."
    )

if "optimizer_state_dict" not in checkpoint:

    raise KeyError(
        "Optimizer state is missing from checkpoint."
    )

print("\nExisting checkpoint loaded.")

print("Starting epoch:", starting_epoch)
print("Training samples:", checkpoint["training_samples"])


# ==========================================
# 2. PREPARE SAME TRAINING SUBSET
# ==========================================

metadata = train_dataset.metadata

positive_indices = metadata.index[
    metadata["Target"] == 1
].tolist()

negative_indices = metadata.index[
    metadata["Target"] == 0
].tolist()

if (
    len(positive_indices) < SAMPLES_PER_CLASS
    or len(negative_indices) < SAMPLES_PER_CLASS
):

    raise ValueError(
        "Insufficient training samples."
    )

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

selected_indices = (
    selected_positive + selected_negative
)

rng.shuffle(selected_indices)

training_subset = Subset(
    train_dataset,
    selected_indices
)

if len(training_subset) != checkpoint["training_samples"]:

    raise ValueError(
        "Training subset size does not match checkpoint."
    )

train_loader = DataLoader(
    training_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=detection_collate_fn
)

print("\nTraining subset verified.")
print("Positive images:", len(selected_positive))
print("Negative images:", len(selected_negative))
print("Total images:", len(training_subset))
print("Training batches:", len(train_loader))


# ==========================================
# 3. REBUILD MODEL ARCHITECTURE
# ==========================================

print("\nRebuilding Faster R-CNN model...")

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

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

print("Existing model weights restored.")


# ==========================================
# 4. RESTORE TRAINABLE PARAMETERS
# ==========================================

for parameter in model.backbone.parameters():

    parameter.requires_grad = False

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.SGD(
    trainable_parameters,
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

print("\nBackbone: Frozen")
print("RPN: Trainable")
print("ROI detection head: Trainable")
print("Optimizer state: Restored")


# ==========================================
# 5. TRAINING HISTORY
# ==========================================

epoch_history = [
    {
        "epoch": starting_epoch,
        "average_loss": checkpoint.get("average_loss"),
        "training_time_seconds": None
    }
]

resumed_training_time = 0.0


# ==========================================
# 6. RESUME TRAINING
# ==========================================

for epoch in range(
    starting_epoch,
    TARGET_EPOCHS
):

    current_epoch = epoch + 1

    model.train()

    # Keep frozen backbone in evaluation mode
    model.backbone.eval()

    running_loss = 0.0

    epoch_start = time.perf_counter()

    print("\n" + "-" * 60)
    print(
        f"EPOCH {current_epoch}/{TARGET_EPOCHS}"
    )
    print("-" * 60)

    for batch_index, (images, targets) in enumerate(
        train_loader
    ):

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

        total_loss = sum(
            loss for loss in loss_dict.values()
        )

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
                f"Epoch {current_epoch} | "
                f"Batch {batch_index + 1}/"
                f"{len(train_loader)} | "
                f"Loss: {total_loss.item():.4f}"
            )

    average_loss = (
        running_loss / len(train_loader)
    )

    epoch_time = (
        time.perf_counter() - epoch_start
    )

    resumed_training_time += epoch_time

    epoch_history.append(
        {
            "epoch": current_epoch,
            "average_loss": average_loss,
            "training_time_seconds": epoch_time
        }
    )

    print("\nEpoch completed.")

    print(
        "Average loss:",
        round(average_loss, 4)
    )

    print(
        "Epoch time:",
        round(epoch_time, 2),
        "seconds"
    )


    # ======================================
    # SAVE EPOCH CHECKPOINT
    # ======================================

    epoch_checkpoint = (
        MODEL_DIR /
        f"fasterrcnn_mobilenetv3_epoch_{current_epoch}.pth"
    )

    torch.save(
        {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "average_loss": average_loss,
            "image_size": 512,
            "training_samples": len(training_subset)
        },
        epoch_checkpoint
    )

    print(
        "Checkpoint saved:",
        epoch_checkpoint.name
    )


# ==========================================
# 7. SAVE FINAL TRAINED MODEL
# ==========================================

torch.save(
    {
        "epoch": TARGET_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_history": epoch_history,
        "image_size": 512,
        "training_samples": len(training_subset)
    },
    FINAL_MODEL_PATH
)


# ==========================================
# 8. SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(epoch_history)

history_df.to_csv(
    MODEL_DIR / "training_history.csv",
    index=False
)


# ==========================================
# 9. FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("STEP 16.7 - TRAINING COMPLETED")
print("=" * 60)

print("\nTraining history:")

print(
    history_df.to_string(index=False)
)

print(
    "\nResumed training time:",
    round(resumed_training_time, 2),
    "seconds"
)

print("\nFinal model saved:")
print(FINAL_MODEL_PATH)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")