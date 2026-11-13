
import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent
evaluation_dir = project_dir / "evaluation"

epoch_1_file = evaluation_dir / "evaluation_metrics.json"
epoch_3_file = evaluation_dir / "evaluation_metrics_epoch_3.json"

with open(epoch_1_file, "r") as file:
    epoch_1 = json.load(file)

with open(epoch_3_file, "r") as file:
    epoch_3 = json.load(file)

metrics = [
    "precision",
    "recall",
    "mean_IoU_of_matched_detections",
    "AP@0.50",
    "mAP@0.50:0.95",
    "true_positives",
    "false_positives",
    "false_negatives",
    "average_inference_time_per_image"
]

comparison = {}

print("\nSTEP 16.9 - EPOCH COMPARISON")
print("=" * 75)
print(f"{'Metric':<40} {'Epoch 1':>12} {'Epoch 3':>12}")
print("-" * 75)

for metric in metrics:

    old_value = epoch_1[metric]
    new_value = epoch_3[metric]

    comparison[metric] = {
        "epoch_1": old_value,
        "epoch_3": new_value,
        "change": new_value - old_value
    }

    print(
        f"{metric:<40} "
        f"{old_value:>12.4f} "
        f"{new_value:>12.4f}"
    )

print("=" * 75)

output_file = evaluation_dir / "epoch_1_vs_epoch_3_comparison.json"

with open(output_file, "w") as file:
    json.dump(comparison, file, indent=4)

print("\nComparison report saved:")
print(output_file)

print("\nSTEP 16.9.1 - COMPLETED")
