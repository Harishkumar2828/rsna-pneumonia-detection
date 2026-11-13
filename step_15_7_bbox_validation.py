
import pandas as pd

# Step 15.7 - Bounding Box Boundary Validation

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Original image dimensions
image_width = 1024
image_height = 1024

# Select annotations containing bounding boxes
bbox_data = annotations.dropna(
    subset=["x", "y", "width", "height"]
).copy()

# Calculate bottom-right coordinates
bbox_data["x2"] = bbox_data["x"] + bbox_data["width"]
bbox_data["y2"] = bbox_data["y"] + bbox_data["height"]

# Validate bounding boxes
valid_boxes = (
    (bbox_data["x"] >= 0) &
    (bbox_data["y"] >= 0) &
    (bbox_data["width"] > 0) &
    (bbox_data["height"] > 0) &
    (bbox_data["x2"] <= image_width) &
    (bbox_data["y2"] <= image_height)
)

# Separate valid and invalid boxes
valid_annotations = bbox_data[valid_boxes]
invalid_annotations = bbox_data[~valid_boxes]

# Display results
print("\n===== STEP 15.7: BOUNDING BOX VALIDATION =====")

print("\nOriginal Image Dimensions:")
print("Width:", image_width)
print("Height:", image_height)

print("\nAnnotation Statistics:")
print("Total bounding boxes:", len(bbox_data))
print("Valid bounding boxes:", len(valid_annotations))
print("Invalid bounding boxes:", len(invalid_annotations))

print("\nValidation Result:")
if len(invalid_annotations) == 0:
    print("All bounding boxes are within image boundaries.")
else:
    print("Some bounding boxes are outside image boundaries.")
    print("\nInvalid Bounding Boxes:")
    print(invalid_annotations[["patientId", "x", "y", "width", "height"]].head())

print("\n==============================================")
