
import pandas as pd
import matplotlib.pyplot as plt

# Step 15.14 - Aspect Ratio Distribution Visualization

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

# Create histogram
plt.figure(figsize=(9, 5))

plt.hist(
    bbox_data["aspect_ratio"],
    bins=40,
    edgecolor="black"
)

# Reference line for square boxes
plt.axvline(
    x=1,
    linestyle="--",
    label="Aspect Ratio = 1"
)

plt.title("RSNA Pneumonia Bounding Box Aspect Ratio Distribution")
plt.xlabel("Aspect Ratio (Width / Height)")
plt.ylabel("Number of Bounding Boxes")
plt.legend()

plt.tight_layout()

# Save chart
plt.savefig("bbox_aspect_ratio_distribution.png", dpi=300)

plt.show()

print("\n===== STEP 15.14 COMPLETED =====")
print("Total Bounding Boxes:", len(bbox_data))
print("Chart saved as bbox_aspect_ratio_distribution.png")
