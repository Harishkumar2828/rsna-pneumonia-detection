
import pandas as pd

# Step 15.6 - Bounding Box Coordinate Calculations

# Load annotation dataset
csv_path = "data/raw/stage_2_train_labels.csv"
annotations = pd.read_csv(csv_path)

# Select the first row containing a bounding box
bbox_data = annotations.dropna(subset=["x", "y", "width", "height"])
annotation = bbox_data.iloc[0]

# Extract coordinates
patient_id = annotation["patientId"]
x = annotation["x"]
y = annotation["y"]
width = annotation["width"]
height = annotation["height"]

# Calculate bottom-right coordinates
x2 = x + width
y2 = y + height

# Display results
print("\n===== STEP 15.6: BOUNDING BOX COORDINATES =====")

print("\nPatient ID:", patient_id)

print("\nOriginal Bounding Box:")
print("X:", x)
print("Y:", y)
print("Width:", width)
print("Height:", height)

print("\nCalculated Coordinates:")
print("Top-left corner:", (x, y))
print("Bottom-right corner:", (x2, y2))

print("\nBounding Box Area:", width * height)
print("\n==============================================")
