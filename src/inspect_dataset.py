import pandas as pd

# Load the training labels
labels = pd.read_csv("data/raw/stage_2_train_labels.csv")

# Load detailed class information
class_info = pd.read_csv("data/raw/stage_2_detailed_class_info.csv")

# Display dataset information
print("TRAINING LABELS")
print(labels.head())

print("\nDETAILED CLASS INFORMATION")
print(class_info.head())

print("\nDataset shape:", labels.shape)
print("Class information shape:", class_info.shape)

print("\nColumn names:", labels.columns.tolist())

# Count total rows
print("\nTOTAL ROWS:", len(labels))

# Count unique patients
print("UNIQUE PATIENTS:", labels["patientId"].nunique())

# Create one record per patient
patient_labels = labels.groupby("patientId")["Target"].max()

# Count patients in each class
print("\nPATIENT-LEVEL DISTRIBUTION:")
print(patient_labels.value_counts().sort_index())

# Calculate percentages
print("\nPATIENT-LEVEL PERCENTAGES:")
print(patient_labels.value_counts(normalize=True).sort_index() * 100)

# Count pneumonia bounding boxes
print("\nTOTAL PNEUMONIA BOUNDING BOXES:")
print(labels[labels["Target"] == 1].shape[0])

# Detailed medical class distribution
print("\nDETAILED CLASS DISTRIBUTION:")
print(class_info["class"].value_counts())

# Check positive patients with multiple bounding boxes
positive_boxes = labels[labels["Target"] == 1]

boxes_per_patient = positive_boxes.groupby("patientId").size()

print("\nPOSITIVE PATIENTS WITH MULTIPLE BOXES:")
print("Patients with 1 box:", (boxes_per_patient == 1).sum())
print("Patients with multiple boxes:", (boxes_per_patient > 1).sum())
print("Maximum boxes for one patient:", boxes_per_patient.max())