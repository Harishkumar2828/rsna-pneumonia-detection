
import pandas as pd

# Step 15.9 - Bounding Box Distribution Analysis

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select annotations containing bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Calculate bounding box area
bbox_data["area"] = bbox_data["width"] * bbox_data["height"]

# Categorize bounding boxes by area
def categorize_size(area):
    if area < 50000:
        return "Small"
    elif area < 100000:
        return "Medium"
    else:
        return "Large"

bbox_data["size_category"] = bbox_data["area"].apply(categorize_size)

# Count bounding boxes in each category
distribution = bbox_data["size_category"].value_counts()

# Calculate percentages
percentages = (
    bbox_data["size_category"].value_counts(normalize=True) * 100
)

# Display results
print("\n===== STEP 15.9: BOUNDING BOX DISTRIBUTION =====")

print("\nTotal Bounding Boxes:", len(bbox_data))

print("\nSIZE CATEGORY DISTRIBUTION")

for category in ["Small", "Medium", "Large"]:
    count = distribution.get(category, 0)
    percentage = percentages.get(category, 0)

    print(f"{category}: {count} ({percentage:.2f}%)")

print("\n===============================================")
