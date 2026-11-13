
import pandas as pd

# Step 15.11 - Bounding Box Position Analysis

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select annotations containing bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Calculate bounding box centre coordinates
bbox_data["center_x"] = bbox_data["x"] + bbox_data["width"] / 2
bbox_data["center_y"] = bbox_data["y"] + bbox_data["height"] / 2

# Divide image into four regions
def identify_region(row):
    if row["center_x"] < 512 and row["center_y"] < 512:
        return "Top-Left"
    elif row["center_x"] >= 512 and row["center_y"] < 512:
        return "Top-Right"
    elif row["center_x"] < 512 and row["center_y"] >= 512:
        return "Bottom-Left"
    else:
        return "Bottom-Right"

bbox_data["region"] = bbox_data.apply(identify_region, axis=1)

# Calculate region distribution
region_counts = bbox_data["region"].value_counts()

# Display results
print("\n===== STEP 15.11: BOUNDING BOX POSITION ANALYSIS =====")

print("\nTotal Bounding Boxes:", len(bbox_data))

print("\nAverage Position:")
print("Average X coordinate:", round(bbox_data["x"].mean(), 2))
print("Average Y coordinate:", round(bbox_data["y"].mean(), 2))
print("Average centre X:", round(bbox_data["center_x"].mean(), 2))
print("Average centre Y:", round(bbox_data["center_y"].mean(), 2))

print("\nREGION DISTRIBUTION")

for region in ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]:
    count = region_counts.get(region, 0)
    percentage = (count / len(bbox_data)) * 100

    print(f"{region}: {count} ({percentage:.2f}%)")

print("\n======================================================")
