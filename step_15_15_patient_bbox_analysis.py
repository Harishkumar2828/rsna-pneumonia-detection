
import pandas as pd

# Step 15.15 - Patient-Level Bounding Box Analysis

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select positive annotations containing bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Count bounding boxes per patient
patient_counts = bbox_data.groupby("patientId").size()

# Calculate statistics
single_box_patients = (patient_counts == 1).sum()
multiple_box_patients = (patient_counts > 1).sum()

# Display results
print("\n===== STEP 15.15: PATIENT-LEVEL ANALYSIS =====")

print("\nTotal Bounding Boxes:", len(bbox_data))
print("Unique Patients with Bounding Boxes:", len(patient_counts))

print("\nPATIENT DISTRIBUTION")
print("Patients with 1 box:", single_box_patients)
print("Patients with multiple boxes:", multiple_box_patients)

print("\nBOX COUNT STATISTICS")
print("Minimum boxes per patient:", patient_counts.min())
print("Maximum boxes per patient:", patient_counts.max())
print("Average boxes per patient:", round(patient_counts.mean(), 2))
print("Median boxes per patient:", patient_counts.median())

print("\n==============================================")
