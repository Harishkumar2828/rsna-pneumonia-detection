
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# STEP 16.1 - DATASET PREPARATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
METADATA_DIR = PROCESSED_DIR / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 16.1 - DATASET PREPARATION")
print("=" * 60)

# 1. Locate CSV files
labels_file = RAW_DIR / "stage_2_train_labels.csv"
class_file = RAW_DIR / "stage_2_detailed_class_info.csv"

if not labels_file.exists():
    raise FileNotFoundError(f"Missing file: {labels_file}")

if not class_file.exists():
    raise FileNotFoundError(f"Missing file: {class_file}")

# 2. Locate training image directory
possible_dirs = [
    PROCESSED_DIR / "train_images",
    RAW_DIR / "stage_2_train_images",
    RAW_DIR / "train_images",
    RAW_DIR
]

IMAGE_DIR = None

for folder in possible_dirs:
    if folder.is_dir() and any(folder.glob("*.dcm")):
        IMAGE_DIR = folder
        break

if IMAGE_DIR is None:
    raise FileNotFoundError(
        "Training DICOM images could not be located."
    )

print(f"\nTraining image directory: {IMAGE_DIR}")

# 3. Load CSV files
labels = pd.read_csv(labels_file)
class_info = pd.read_csv(class_file)

print("\nLoaded datasets:")
print("Training annotation rows:", len(labels))
print("Class information rows:", len(class_info))

# 4. Validate required columns
required_label_columns = {
    "patientId", "x", "y", "width", "height", "Target"
}

required_class_columns = {"patientId", "class"}

if not required_label_columns.issubset(labels.columns):
    raise ValueError("Required annotation columns are missing.")

if not required_class_columns.issubset(class_info.columns):
    raise ValueError("Required class information columns are missing.")

# Check whether duplicate patient IDs have conflicting classes

class_conflicts = (
    class_info.groupby("patientId")["class"]
    .nunique()
)

conflicting_patients = class_conflicts[class_conflicts > 1]

if len(conflicting_patients) > 0:
    raise ValueError(
        f"Conflicting class labels found for "
        f"{len(conflicting_patients)} patients."
    )

# Keep one class record per patient
class_info = class_info.drop_duplicates(
    subset="patientId"
).copy()

print("\nClass information after removing repeated patient records:")
print("Unique patients:", len(class_info))
print("Class distribution:")
print(class_info["class"].value_counts())

# 5. Create patient-level metadata
patient_targets = (
    labels.groupby("patientId")
    .agg(
        Target=("Target", "max"),
        num_boxes=("Target", "sum")
    )
    .reset_index()
)

patient_metadata = patient_targets.merge(
    class_info[["patientId", "class"]],
    on="patientId",
    how="left",
    validate="one_to_one"
)

if patient_metadata["class"].isna().any():
    raise ValueError("Some patients have missing class labels.")

patient_metadata["Target"] = patient_metadata["Target"].astype(int)
patient_metadata["num_boxes"] = patient_metadata["num_boxes"].astype(int)

# 6. Add image paths
patient_metadata["image_path"] = patient_metadata["patientId"].apply(
    lambda patient_id: str(
        (IMAGE_DIR / f"{patient_id}.dcm").relative_to(PROJECT_DIR)
    )
)

# 7. Validate image files
print("\nValidating training images...")

missing_images = [
    patient_id
    for patient_id in patient_metadata["patientId"]
    if not (IMAGE_DIR / f"{patient_id}.dcm").is_file()
]

if missing_images:
    print("Missing image count:", len(missing_images))
    print("Examples:", missing_images[:5])
    raise FileNotFoundError("Some training images are missing.")

print("All patient image files exist.")

# 8. Save complete patient metadata
patient_metadata.to_csv(
    METADATA_DIR / "all_patient_metadata.csv",
    index=False
)

# 9. Create patient-level 80:20 split
train_patients, val_patients = train_test_split(
    patient_metadata,
    test_size=0.20,
    random_state=42,
    stratify=patient_metadata["class"]
)

train_ids = set(train_patients["patientId"])
val_ids = set(val_patients["patientId"])

# 10. Preserve all bounding-box annotations
annotations = labels.merge(
    class_info[["patientId", "class"]],
    on="patientId",
    how="left",
    validate="many_to_one"
)

annotations["image_path"] = annotations["patientId"].apply(
    lambda patient_id: str(
        (IMAGE_DIR / f"{patient_id}.dcm").relative_to(PROJECT_DIR)
    )
)

train_annotations = annotations[
    annotations["patientId"].isin(train_ids)
].copy()

val_annotations = annotations[
    annotations["patientId"].isin(val_ids)
].copy()

# 11. Save training and validation metadata
train_patients.to_csv(
    METADATA_DIR / "train_metadata.csv",
    index=False
)

val_patients.to_csv(
    METADATA_DIR / "validation_metadata.csv",
    index=False
)

train_annotations.to_csv(
    METADATA_DIR / "train_annotations.csv",
    index=False
)

val_annotations.to_csv(
    METADATA_DIR / "validation_annotations.csv",
    index=False
)

# 12. Final validation
assert train_ids.isdisjoint(val_ids), "Patient leakage detected!"

assert len(train_patients) + len(val_patients) == len(patient_metadata)

assert len(train_annotations) + len(val_annotations) == len(annotations)

# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print("\nPATIENT DISTRIBUTION")
print("Total patients:", len(patient_metadata))
print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))

print("\nCLASS DISTRIBUTION - TRAINING")
print(train_patients["class"].value_counts())

print("\nCLASS DISTRIBUTION - VALIDATION")
print(val_patients["class"].value_counts())

print("\nANNOTATION DISTRIBUTION")
print("Total annotation rows:", len(annotations))
print("Training annotation rows:", len(train_annotations))
print("Validation annotation rows:", len(val_annotations))

print("\nOUTPUT FILES")
for file in sorted(METADATA_DIR.glob("*.csv")):
    print(file.name)

print("\nPatient leakage check: PASSED")
print("Image validation: PASSED")
print("\nSTEP 16.1 COMPLETED SUCCESSFULLY!")
