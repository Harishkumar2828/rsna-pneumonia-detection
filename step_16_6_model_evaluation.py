
import json
import time
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from step_16_2_image_dataset import (
    val_dataset,
    detection_collate_fn
)

# ==========================================
# STEP 16.6 - MODEL EVALUATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
EVALUATION_DIR = PROJECT_DIR / "evaluation"

CHECKPOINT_PATH = (
    MODEL_DIR / "fasterrcnn_mobilenetv3_trained.pth"
)

DEVICE = torch.device("cpu")

BATCH_SIZE = 2
CONFIDENCE_THRESHOLD = 0.50
AP_SCORE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.50

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)

# ==========================================
# 1. IOU CALCULATION
# ==========================================

def calculate_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )

    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union

# ==========================================
# 2. DETECTION MATCHING
# ==========================================

def match_detections(
    predictions,
    ground_truths,
    iou_threshold,
    score_threshold=None
):

    total_ground_truths = sum(
        len(boxes) for boxes in ground_truths
    )

    matched_ground_truths = [
        set() for _ in ground_truths
    ]

    sorted_predictions = sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True
    )

    true_positives = 0
    false_positives = 0
    matched_ious = []

    for prediction in sorted_predictions:

        if (
            score_threshold is not None
            and prediction["score"] < score_threshold
        ):
            continue

        image_index = prediction["image_index"]
        predicted_box = prediction["box"]

        image_ground_truths = ground_truths[
            image_index
        ]

        if len(image_ground_truths) == 0:

            false_positives += 1
            continue

        best_iou = 0.0
        best_index = -1

        for gt_index, gt_box in enumerate(
            image_ground_truths
        ):

            if gt_index in matched_ground_truths[
                image_index
            ]:
                continue

            current_iou = calculate_iou(
                predicted_box,
                gt_box
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_index = gt_index

        if (
            best_index >= 0
            and best_iou >= iou_threshold
        ):

            true_positives += 1

            matched_ground_truths[
                image_index
            ].add(best_index)

            matched_ious.append(best_iou)

        else:

            false_positives += 1

    false_negatives = (
        total_ground_truths - true_positives
    )

    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "matched_ious": matched_ious
    }

# ==========================================
# 3. AVERAGE PRECISION CALCULATION
# ==========================================

def calculate_average_precision(
    predictions,
    ground_truths,
    iou_threshold
):

    total_ground_truths = sum(
        len(boxes) for boxes in ground_truths
    )

    if total_ground_truths == 0:
        return 0.0

    sorted_predictions = sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True
    )

    matched_ground_truths = [
        set() for _ in ground_truths
    ]

    tp_list = []
    fp_list = []

    for prediction in sorted_predictions:

        image_index = prediction["image_index"]
        predicted_box = prediction["box"]

        image_ground_truths = ground_truths[
            image_index
        ]

        best_iou = 0.0
        best_index = -1

        for gt_index, gt_box in enumerate(
            image_ground_truths
        ):

            if gt_index in matched_ground_truths[
                image_index
            ]:
                continue

            current_iou = calculate_iou(
                predicted_box,
                gt_box
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_index = gt_index

        if (
            best_index >= 0
            and best_iou >= iou_threshold
        ):

            tp_list.append(1)
            fp_list.append(0)

            matched_ground_truths[
                image_index
            ].add(best_index)

        else:

            tp_list.append(0)
            fp_list.append(1)

    if len(tp_list) == 0:
        return 0.0

    tp_cumulative = np.cumsum(tp_list)
    fp_cumulative = np.cumsum(fp_list)

    recalls = (
        tp_cumulative / total_ground_truths
    )

    precisions = (
        tp_cumulative /
        (tp_cumulative + fp_cumulative)
    )

    # Precision envelope

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(
        ([0.0], precisions, [0.0])
    )

    for i in range(len(precisions) - 2, -1, -1):

        precisions[i] = max(
            precisions[i],
            precisions[i + 1]
        )

    change_indices = np.where(
        recalls[1:] != recalls[:-1]
    )[0]

    average_precision = np.sum(
        (recalls[change_indices + 1] -
         recalls[change_indices]) *
        precisions[change_indices + 1]
    )

    return float(average_precision)

# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

print("=" * 60)
print("STEP 16.6 - MODEL EVALUATION")
print("=" * 60)

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT_PATH}"
    )

print("\nLoading trained model...")

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

model.roi_heads.score_thresh = AP_SCORE_THRESHOLD

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

print("Checkpoint epoch:", checkpoint.get("epoch"))
print(
    "Checkpoint training samples:",
    checkpoint.get("training_samples")
)

# ==========================================
# 5. PREPARE UNTOUCHED VALIDATION DATASET
# ==========================================

print("\nPreparing validation DataLoader...")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=detection_collate_fn
)

print("Validation patients:", len(val_dataset))
print("Validation batches:", len(val_loader))

# ==========================================
# 6. GENERATE MODEL PREDICTIONS
# ==========================================

print("\nRunning validation inference...")

all_predictions = []
all_ground_truths = []

total_inference_time = 0.0

with torch.inference_mode():

    for batch_index, (images, targets) in enumerate(
        val_loader
    ):

        images = [
            image.to(DEVICE)
            for image in images
        ]

        start_time = time.perf_counter()

        outputs = model(images)

        total_inference_time += (
            time.perf_counter() - start_time
        )

        for image_index, (output, target) in enumerate(
            zip(outputs, targets)
        ):

            global_image_index = (
                len(all_ground_truths)
            )

            gt_boxes = (
                target["boxes"]
                .cpu()
                .numpy()
                .reshape(-1, 4)
            )

            all_ground_truths.append(gt_boxes)

            pred_boxes = (
                output["boxes"]
                .cpu()
                .numpy()
            )

            pred_scores = (
                output["scores"]
                .cpu()
                .numpy()
            )

            pred_labels = (
                output["labels"]
                .cpu()
                .numpy()
            )

            for box, score, label in zip(
                pred_boxes,
                pred_scores,
                pred_labels
            ):

                if label != 1:
                    continue

                all_predictions.append({
                    "image_index": global_image_index,
                    "box": box,
                    "score": float(score)
                })

        if (
            (batch_index + 1) % 100 == 0
            or batch_index + 1 == len(val_loader)
        ):

            print(
                f"Processed batch "
                f"{batch_index + 1}/{len(val_loader)}"
            )

# ==========================================
# 7. DATASET SUMMARY
# ==========================================

total_images = len(all_ground_truths)

total_gt_boxes = sum(
    len(boxes) for boxes in all_ground_truths
)

positive_images = sum(
    len(boxes) > 0 for boxes in all_ground_truths
)

negative_images = (
    total_images - positive_images
)

print("\nValidation dataset summary:")

print("Total images:", total_images)
print("Positive images:", positive_images)
print("Negative images:", negative_images)
print("Ground-truth boxes:", total_gt_boxes)
print("Predicted boxes:", len(all_predictions))

if total_gt_boxes == 0:

    raise ValueError(
        "No ground-truth boxes found in validation dataset."
    )

# ==========================================
# 8. PRECISION, RECALL AND IOU
# ==========================================

print("\nCalculating detection metrics...")

detection_stats = match_detections(
    all_predictions,
    all_ground_truths,
    IOU_THRESHOLD,
    CONFIDENCE_THRESHOLD
)

tp = detection_stats["TP"]
fp = detection_stats["FP"]
fn = detection_stats["FN"]

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)

mean_iou = (
    float(np.mean(detection_stats["matched_ious"]))
    if detection_stats["matched_ious"]
    else 0.0
)

# ==========================================
# 9. MAP CALCULATION
# ==========================================

print("\nCalculating Average Precision...")

iou_thresholds = np.arange(
    0.50,
    0.96,
    0.05
)

ap_results = {}

for threshold in iou_thresholds:

    ap = calculate_average_precision(
        all_predictions,
        all_ground_truths,
        float(threshold)
    )

    ap_results[f"AP@{threshold:.2f}"] = ap

    print(
        f"AP @ IoU {threshold:.2f}: {ap:.4f}"
    )

map_50 = ap_results["AP@0.50"]

map_50_95 = float(
    np.mean(list(ap_results.values()))
)

# ==========================================
# 10. SAVE EVALUATION RESULTS
# ==========================================

metrics = {

    "model": "Faster R-CNN MobileNetV3 320 FPN",

    "evaluation_dataset": "Untouched validation dataset",

    "validation_images": total_images,

    "positive_images": positive_images,

    "negative_images": negative_images,

    "ground_truth_boxes": total_gt_boxes,

    "predicted_boxes": len(all_predictions),

    "confidence_threshold": CONFIDENCE_THRESHOLD,

    "IoU_threshold": IOU_THRESHOLD,

    "true_positives": tp,

    "false_positives": fp,

    "false_negatives": fn,

    "mean_IoU_of_matched_detections": mean_iou,

    "precision": precision,

    "recall": recall,

    "AP@0.50": map_50,

    "mAP@0.50:0.95": map_50_95,

    "inference_time_seconds": total_inference_time,

    "average_inference_time_per_image": (
        total_inference_time / total_images
    ),

    **ap_results
}

metrics_path = (
    EVALUATION_DIR / "evaluation_metrics_epoch_3.json"
)

with open(metrics_path, "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

# ==========================================
# 11. FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("STEP 16.6 - EVALUATION RESULTS")
print("=" * 60)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Mean IoU of matched detections: {mean_iou:.4f}")
print(f"AP@0.50: {map_50:.4f}")
print(f"mAP@0.50:0.95: {map_50_95:.4f}")

print("\nDetection counts:")

print("True Positives:", tp)
print("False Positives:", fp)
print("False Negatives:", fn)

print("\nInference time:", round(
    total_inference_time, 2
), "seconds")

print("Average inference time per image:", round(
    total_inference_time / total_images, 4
), "seconds")

print("\nEvaluation report saved:")
print(metrics_path)

print("\n" + "=" * 60)
print("STEP 16.6 - EVALUATION COMPLETED")
print("=" * 60)