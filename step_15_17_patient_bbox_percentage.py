
import pandas as pd

# Step 15.17 - Patient Bounding Box Percentage Analysis

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select annotations containing bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Count boxes per patient
patient_counts = bbox_data.groupby("patientId").size()

# Calculate distribution
distribution = patient_counts.value_counts().sort_index()

# Calculate percentages
percentages = (
    distribution / len(patient_counts) * 100
)

# Display results
print("\n===== STEP 15.17: PATIENT BBOX PERCENTAGE =====")

print("Total Patients:", len(patient_counts))

print("\nBOX COUNT DISTRIBUTION")

for boxes, count in distribution.items():
    percentage = percentages[boxes]

    print(
        f"{boxes} box(es): {count} patients "
        f"({percentage:.2f}%)"
    )

print("\nTotal Patients:", distribution.sum())
print("Total Percentage:", round(percentages.sum(), 2), "%")

print("\n==============================================")
