
import torch
import numpy as np
import time

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
# STEP 16.4 - TRAINING BENCHMARK
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"

INITIAL_MODEL = MODEL_DIR / "fasterrcnn_mobilenetv3_initial.pth"

OUTPUT_MODEL = MODEL_DIR / "fasterrcnn_mobilenetv3_benchmark.pth"

DEVICE = torch.device("cpu")

BATCH_SIZE = 2
EPOCHS = 1
SAMPLES_PER_CLASS = 16

print("\n" + "=" * 60)
print("STEP 16.4 - TRAINING BENCHMARK")
print("=" * 60)


# ==========================================
# 1. SELECT BALANCED TRAINING SUBSET
# ==========================================

metadata = train_dataset.metadata

positive_indices = metadata.index[
    metadata["Target"] == 1
].tolist()

negative_indices = metadata.index[
    metadata["Target"] == 0
].tolist()

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

benchmark_dataset = Subset(
    train_dataset,
    selected_indices
)

benchmark_loader = DataLoader(
    benchmark_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=detection_collate_fn
)

print("\nTraining subset prepared.")
print("Positive images:", len(selected_positive))
print("Negative images:", len(selected_negative))
print("Total images:", len(benchmark_dataset))
print("Training batches:", len(benchmark_loader))


# ==========================================
# 2. INITIALIZE MODEL ARCHITECTURE
# ==========================================

print("\nLoading initialized model...")

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

# Load the pretrained checkpoint from Step 16.3
checkpoint = torch.load(
    INITIAL_MODEL,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)

print("Initial model checkpoint loaded.")


# ==========================================
# 3. FREEZE BACKBONE AND RPN
# ==========================================

for parameter in model.parameters():
    parameter.requires_grad = False

# Train ROI box head
for parameter in model.roi_heads.box_head.parameters():
    parameter.requires_grad = True

# Train classification and box prediction head
for parameter in model.roi_heads.box_predictor.parameters():
    parameter.requires_grad = True

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

print("\nTrainable parameters:",
      sum(p.numel() for p in trainable_parameters))

print("Backbone: Frozen")
print("RPN: Frozen")
print("ROI box head: Trainable")
print("Prediction head: Trainable")


# ==========================================
# 4. OPTIMIZER
# ==========================================

optimizer = torch.optim.SGD(
    trainable_parameters,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)


# ==========================================
# 5. TRAINING LOOP
# ==========================================

model.train()

# Keep frozen feature extractor and RPN in evaluation mode
model.backbone.eval()
model.rpn.eval()

print("\nStarting benchmark training...")

start_time = time.perf_counter()

epoch_losses = []

for epoch in range(EPOCHS):

    running_loss = 0.0

    for batch_index, (images, targets) in enumerate(
        benchmark_loader
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
            raise ValueError("Non-finite loss detected.")

        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Batch {batch_index + 1}/{len(benchmark_loader)} | "
            f"Loss: {total_loss.item():.4f}"
        )

    average_loss = running_loss / len(benchmark_loader)

    epoch_losses.append(average_loss)

    print(
        f"\nEpoch {epoch + 1} Average Loss: "
        f"{average_loss:.4f}"
    )


# ==========================================
# 6. TRAINING TIME
# ==========================================

end_time = time.perf_counter()

training_time = end_time - start_time

print("\nTraining time:", round(training_time, 2), "seconds")


# ==========================================
# 7. SAVE BENCHMARK CHECKPOINT
# ==========================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": EPOCHS,
        "average_loss": epoch_losses[-1],
        "image_size": 512,
        "batch_size": BATCH_SIZE,
        "training_samples": len(benchmark_dataset)
    },
    OUTPUT_MODEL
)

print("\nBenchmark model saved:")
print(OUTPUT_MODEL)


# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("STEP 16.4 COMPLETED")
print("=" * 60)

print("Training images:", len(benchmark_dataset))
print("Epochs:", EPOCHS)
print("Final average loss:", round(epoch_losses[-1], 4))
print("Training time:", round(training_time, 2), "seconds")

print("Model checkpoint: SAVED")
print("CPU training benchmark: PASSED")