
import pandas as pd
import matplotlib.pyplot as plt

# Step 15.16 - Patient Bounding Box Distribution

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select annotations containing bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Count bounding boxes per patient
patient_counts = bbox_data.groupby("patientId").size()

# Calculate patient distribution
distribution = patient_counts.value_counts().sort_index()

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(
    distribution.index.astype(str),
    distribution.values
)

# Add count labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.title("RSNA Patient-Level Bounding Box Distribution")
plt.xlabel("Number of Bounding Boxes per Patient")
plt.ylabel("Number of Patients")

plt.tight_layout()

# Save chart
plt.savefig("patient_bbox_distribution.png", dpi=300)

plt.show()

print("\n===== STEP 15.16 COMPLETED =====")
print(distribution)
print("\nChart saved as patient_bbox_distribution.png")
