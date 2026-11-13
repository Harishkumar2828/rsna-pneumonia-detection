
import pandas as pd

# Step 15.8 - Bounding Box Size Analysis

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select annotations containing bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Calculate bounding box area
bbox_data["area"] = bbox_data["width"] * bbox_data["height"]

# Calculate bounding box statistics
print("\n===== STEP 15.8: BOUNDING BOX SIZE ANALYSIS =====")

print("\nTotal Bounding Boxes:", len(bbox_data))

print("\nWIDTH ANALYSIS")
print("Minimum width:", bbox_data["width"].min())
print("Maximum width:", bbox_data["width"].max())
print("Average width:", round(bbox_data["width"].mean(), 2))

print("\nHEIGHT ANALYSIS")
print("Minimum height:", bbox_data["height"].min())
print("Maximum height:", bbox_data["height"].max())
print("Average height:", round(bbox_data["height"].mean(), 2))

print("\nAREA ANALYSIS")
print("Minimum area:", bbox_data["area"].min())
print("Maximum area:", bbox_data["area"].max())
print("Average area:", round(bbox_data["area"].mean(), 2))
print("Median area:", round(bbox_data["area"].median(), 2))

print("\n===============================================")
