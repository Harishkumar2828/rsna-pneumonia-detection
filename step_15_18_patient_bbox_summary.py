
import pandas as pd

# Step 15.18 - Patient Bounding Box Summary

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select valid bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Count bounding boxes per patient
patient_counts = (
    bbox_data.groupby("patientId")
    .size()
    .reset_index(name="bbox_count")
)

total_patients = len(patient_counts)
total_boxes = len(bbox_data)

# Calculate each patient's percentage of all boxes
patient_counts["bbox_percentage"] = (
    patient_counts["bbox_count"] / total_boxes * 100
)

# Create distribution summary
summary = (
    patient_counts.groupby("bbox_count")
    .agg(patient_count=("patientId", "count"))
    .reset_index()
)

# Calculate percentage of patients in each category
summary["patient_percentage"] = (
    summary["patient_count"] / total_patients * 100
)

# Save reports
patient_counts.to_csv("patient_bbox_summary.csv", index=False)

summary.to_csv(
    "patient_bbox_distribution_summary.csv",
    index=False
)

# Display results
print("\n===== STEP 15.18 COMPLETED =====")

print("\nPatient Summary:")
print(patient_counts.head())

print("\nCorrected Distribution Summary:")
print(summary.to_string(index=False))

print("\nTotal Patients:", total_patients)
print("Total Bounding Boxes:", total_boxes)
print("Total Patient Percentage:", round(summary["patient_percentage"].sum(), 2), "%")

print("\nGenerated Files:")
print("1. patient_bbox_summary.csv")
print("2. patient_bbox_distribution_summary.csv")
