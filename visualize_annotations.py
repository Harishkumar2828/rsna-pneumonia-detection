import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Step 1: Load annotation CSV
labels = pd.read_csv("data/raw/stage_2_train_labels.csv")

# Step 2: Select our sample image
sample_id = "0ab261f9-4eb5-42ab-a9a5-e918904d6356"

# Step 3: Get annotations for this image
sample_annotations = labels[
    labels["patientId"] == sample_id
]

print("Patient ID:", sample_id)
print("Number of bounding boxes:", len(sample_annotations))

# Step 4: Load DICOM image
image_path = f"data/processed/train_images/{sample_id}.dcm"

dicom_data = pydicom.dcmread(image_path)
image = dicom_data.pixel_array

# Step 5: Display image
fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(image, cmap="gray")

# Step 6: Draw bounding boxes
for _, row in sample_annotations.iterrows():

    rectangle = patches.Rectangle(
        (row["x"], row["y"]),
        row["width"],
        row["height"],
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rectangle)

# Step 7: Final display
ax.set_title("RSNA Pneumonia Bounding Boxes")
ax.axis("off")

plt.show()