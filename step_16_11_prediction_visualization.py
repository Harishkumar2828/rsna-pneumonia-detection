
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Rectangle

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from step_16_2_image_dataset import val_dataset


# ==========================================
# STEP 16.11 - PREDICTION VISUALIZATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
EVALUATION_DIR = PROJECT_DIR / "evaluation"

OUTPUT_DIR = EVALUATION_DIR / "visualizations"

CHECKPOINT_PATH = (
    MODEL_DIR / "fasterrcnn_mobilenetv3_trained.pth"
)

DEVICE = torch.device("cpu")

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

MAX_POSITIVE_IMAGES = 100
MAX_NEGATIVE_IMAGES = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)

print("=" * 60)
print("STEP 16.11 - PREDICTION VISUALIZATION")
print("=" * 60)


# ==========================================
# 1. CALCULATE IOU
# ==========================================

def calculate_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area_a = (
        max(0, box_a[2] - box_a[0]) *
        max(0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0, box_b[2] - box_b[0]) *
        max(0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ==========================================
# 2. MATCH PREDICTIONS WITH GROUND TRUTH
# ==========================================

def analyze_detections(predictions, ground_truths):

    matched_gt = set()
    matched_predictions = []
    false_predictions = []

    predictions = sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True
    )

    for prediction in predictions:

        best_iou = 0.0
        best_index = -1

        for gt_index, gt_box in enumerate(ground_truths):

            if gt_index in matched_gt:
                continue

            current_iou = calculate_iou(
                prediction["box"],
                gt_box
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_index = gt_index

        if best_index >= 0 and best_iou >= IOU_THRESHOLD:

            matched_gt.add(best_index)

            matched_predictions.append(prediction)

        else:

            false_predictions.append(prediction)

    missed_gt = [
        box
        for index, box in enumerate(ground_truths)
        if index not in matched_gt
    ]

    return (
        matched_predictions,
        false_predictions,
        missed_gt
    )


# ==========================================
# 3. LOAD EPOCH 3 MODEL
# ==========================================

print("\nLoading Epoch 3 model...")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

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
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.roi_heads.score_thresh = 0.001

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Checkpoint epoch:", checkpoint.get("epoch"))


# ==========================================
# 4. IDENTIFY POSITIVE AND NEGATIVE IMAGES
# ==========================================

print("\nIdentifying validation images...")

positive_indices = []
negative_indices = []

for index, record in val_dataset.metadata.iterrows():

    patient_id = str(record["patientId"])

    annotations = (
        val_dataset.annotations_by_patient.get(patient_id)
    )

    has_pneumonia = False

    if annotations is not None:

        valid_annotations = annotations.dropna(
            subset=["x", "y", "width", "height"]
        )

        valid_annotations = valid_annotations[
            valid_annotations["Target"] == 1
        ]

        has_pneumonia = len(valid_annotations) > 0

    if has_pneumonia:

        if len(positive_indices) < MAX_POSITIVE_IMAGES:
            positive_indices.append(index)

    else:

        if len(negative_indices) < MAX_NEGATIVE_IMAGES:
            negative_indices.append(index)

    if (
        len(positive_indices) >= MAX_POSITIVE_IMAGES
        and len(negative_indices) >= MAX_NEGATIVE_IMAGES
    ):
        break

print("Positive candidates:", len(positive_indices))
print("Negative candidates:", len(negative_indices))


# ==========================================
# 5. VISUALIZATION FUNCTION
# ==========================================

def save_visualization(
    image,
    patient_id,
    ground_truths,
    matched_predictions,
    false_predictions,
    missed_gt,
    filename
):

    image_array = (
        image[0].cpu().numpy()
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image_array, cmap="gray")

    # Matched ground-truth boxes
    for box in ground_truths:

        is_missed = any(
           np.array_equal(box, missed_box)
           for missed_box in missed_gt
)

        if not is_missed:
            color = "lime"
            label = "Matched GT"
        else:
            color = "yellow"
            label = "Missed GT"

        x1, y1, x2, y2 = box

        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2
            )
        )

        ax.text(
            x1,
            max(0, y1 - 5),
            label,
            color=color,
            fontsize=8,
            backgroundcolor="black"
        )

    # Correct predictions
    for prediction in matched_predictions:

        x1, y1, x2, y2 = prediction["box"]

        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="blue",
                linewidth=2
            )
        )

        ax.text(
            x1,
            y2 + 3,
            f"TP {prediction['score']:.2f}",
            color="blue",
            fontsize=8,
            backgroundcolor="white"
        )

    # Incorrect predictions
    for prediction in false_predictions:

        x1, y1, x2, y2 = prediction["box"]

        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
        )

        ax.text(
            x1,
            y1 + 5,
            f"FP {prediction['score']:.2f}",
            color="red",
            fontsize=8,
            backgroundcolor="white"
        )

    ax.set_title(
        f"Patient: {patient_id}\n"
        f"GT: {len(ground_truths)} | "
        f"TP: {len(matched_predictions)} | "
        f"FP: {len(false_predictions)} | "
        f"FN: {len(missed_gt)}"
    )

    ax.axis("off")

    plt.tight_layout()

    save_path = OUTPUT_DIR / filename
    print("\nDEBUG - IMAGE SAVING")
    print("Filename:", repr(filename))
    print("Filename type:", type(filename))
    print("Output directory:", repr(str(OUTPUT_DIR)))
    print("Save path:", repr(str(save_path)))
    print("Directory exists:", OUTPUT_DIR.exists())
    print("Is directory:", OUTPUT_DIR.is_dir())

    from io import BytesIO

    image_buffer = BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    save_path.write_bytes(image_buffer.getvalue())
    image_buffer.close()

    plt.close(fig)

    print("Saved:", save_path.name)


# ==========================================
# 6. RUN SELECTED VALIDATION INFERENCE
# ==========================================

print("\nRunning prediction analysis...")

found = {
    "true_positive": False,
    "false_positive_positive_image": False,
    "false_negative": False,
    "false_positive_negative_image": False,
    "clean_negative": False
}

positive_scanned = 0
negative_scanned = 0
true_positive_patient_id = None

candidates = (
    [(index, True) for index in positive_indices] +
    [(index, False) for index in negative_indices]
)

for index, is_positive in candidates:

    image, target = val_dataset[index]

    patient_id = str(
        val_dataset.metadata.iloc[index]["patientId"]
    )

    with torch.inference_mode():

        output = model([
            image.to(DEVICE)
        ])[0]

    gt_boxes = (
        target["boxes"].cpu().numpy()
    )

    pred_boxes = (
        output["boxes"].cpu().numpy()
    )

    pred_scores = (
        output["scores"].cpu().numpy()
    )

    pred_labels = (
        output["labels"].cpu().numpy()
    )

    predictions = []

    for box, score, label in zip(
        pred_boxes,
        pred_scores,
        pred_labels
    ):

        if (
            label == 1
            and score >= CONFIDENCE_THRESHOLD
        ):

            predictions.append({
                "box": box,
                "score": float(score)
            })

    matched, false_predictions, missed = analyze_detections(
        predictions,
        gt_boxes
    )

    if is_positive:
        positive_scanned += 1
    else:
        negative_scanned += 1

        # True Positive example
    if (
        len(matched) > 0
        and not found["true_positive"]
    ):

        save_visualization(
            image, patient_id, gt_boxes,
            matched, false_predictions, missed,
            "true_positive_example.png"
        )

        found["true_positive"] = True
        true_positive_patient_id = patient_id

    # False Positive on positive image
    if (
        is_positive
        and len(false_predictions) > 0
        and not found["false_positive_positive_image"]
    ):

        save_visualization(
            image, patient_id, gt_boxes,
            matched, false_predictions, missed,
            "false_positive_positive_image.png"
        )

        found["false_positive_positive_image"] = True

    # False Negative example - use a different patient
    if (
        is_positive
        and len(missed) > 0
        and patient_id != true_positive_patient_id
        and not found["false_negative"]
    ):

        save_visualization(
            image, patient_id, gt_boxes,
            matched, false_predictions, missed,
            "false_negative_example.png"
        )

        found["false_negative"] = True

    # False Positive on negative image
    if (
        not is_positive
        and len(false_predictions) > 0
        and not found["false_positive_negative_image"]
    ):

        save_visualization(
            image, patient_id, gt_boxes,
            matched, false_predictions, missed,
            "false_positive_negative_image.png"
        )

        found["false_positive_negative_image"] = True

    # Clean negative image
    if (
        not is_positive
        and len(predictions) == 0
        and not found["clean_negative"]
    ):

        save_visualization(
            image, patient_id, gt_boxes,
            matched, false_predictions, missed,
            "clean_negative_example.png"
        )

        found["clean_negative"] = True

    if all(found.values()):
        break

    if (
        positive_scanned >= MAX_POSITIVE_IMAGES
        and negative_scanned >= MAX_NEGATIVE_IMAGES
    ):
        break

    if (
        positive_scanned >= MAX_POSITIVE_IMAGES
        and is_positive
    ):
        continue

    if (
        negative_scanned >= MAX_NEGATIVE_IMAGES
        and not is_positive
    ):
        continue


# ==========================================
# 7. FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("STEP 16.11 - VISUALIZATION RESULTS")
print("=" * 60)

print("Positive images scanned:", positive_scanned)
print("Negative images scanned:", negative_scanned)

for category, status in found.items():

    print(
        category,
        ":",
        "FOUND" if status else "NOT FOUND"
    )

print("\nVisualization directory:")
print(OUTPUT_DIR)

print("\nSTEP 16.11 - ANALYSIS COMPLETED")
print("=" * 60)
