
import pandas as pd

# Step 15.13 - Bounding Box Aspect Ratio Analysis

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select valid bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Calculate aspect ratio
bbox_data["aspect_ratio"] = (
    bbox_data["width"] / bbox_data["height"]
)

# Calculate statistics
print("\n===== STEP 15.13: ASPECT RATIO ANALYSIS =====")

print("\nTotal Bounding Boxes:", len(bbox_data))

print("\nASPECT RATIO STATISTICS")
print("Minimum:", round(bbox_data["aspect_ratio"].min(), 3))
print("Maximum:", round(bbox_data["aspect_ratio"].max(), 3))
print("Average:", round(bbox_data["aspect_ratio"].mean(), 3))
print("Median:", round(bbox_data["aspect_ratio"].median(), 3))

# Categorize box shapes
wide = (bbox_data["aspect_ratio"] > 1).sum()
tall = (bbox_data["aspect_ratio"] < 1).sum()
square = (bbox_data["aspect_ratio"] == 1).sum()

print("\nBOX SHAPE DISTRIBUTION")
print("Wide boxes:", wide)
print("Tall boxes:", tall)
print("Square boxes:", square)

print("\n============================================")
