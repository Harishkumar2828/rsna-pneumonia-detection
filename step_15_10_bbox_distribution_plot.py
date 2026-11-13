
import pandas as pd
import matplotlib.pyplot as plt

# Step 15.10 - Bounding Box Distribution Visualization

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select annotations containing bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Calculate bounding box area
bbox_data["area"] = bbox_data["width"] * bbox_data["height"]

# Categorize bounding boxes
def categorize_size(area):
    if area < 50000:
        return "Small"
    elif area < 100000:
        return "Medium"
    else:
        return "Large"

bbox_data["size_category"] = bbox_data["area"].apply(categorize_size)

# Calculate category counts
distribution = (
    bbox_data["size_category"]
    .value_counts()
    .reindex(["Small", "Medium", "Large"], fill_value=0)
)

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(
    distribution.index,
    distribution.values
)

# Add count labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

plt.title("RSNA Pneumonia Bounding Box Size Distribution")
plt.xlabel("Bounding Box Category")
plt.ylabel("Number of Bounding Boxes")
plt.tight_layout()

# Save and display chart
plt.savefig("bbox_distribution.png", dpi=300)
plt.show()

print("\n===== STEP 15.10 COMPLETED =====")
print(distribution)
print("\nChart saved as bbox_distribution.png")
