
import pandas as pd
import matplotlib.pyplot as plt

# Step 15.12 - Bounding Box Position Visualization

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

# Create scatter plot
plt.figure(figsize=(9, 8))

plt.scatter(
    bbox_data["center_x"],
    bbox_data["center_y"],
    alpha=0.3,
    s=10
)

# Image boundaries
plt.xlim(0, 1024)
plt.ylim(1024, 0)

# Divide image into four regions
plt.axvline(x=512, linestyle="--")
plt.axhline(y=512, linestyle="--")

# Add labels
plt.title("RSNA Pneumonia Bounding Box Position Distribution")
plt.xlabel("Horizontal Position (X)")
plt.ylabel("Vertical Position (Y)")

plt.tight_layout()

# Save visualization
plt.savefig("bbox_position_distribution.png", dpi=300)

plt.show()

print("\n===== STEP 15.12 COMPLETED =====")
print("Total Bounding Boxes:", len(bbox_data))
print("Chart saved as bbox_position_distribution.png")
