
import pydicom
import matplotlib.pyplot as plt

# Step 1: Read the DICOM image
path = "data/processed/train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm"

ds = pydicom.dcmread(path)

# Step 2: Extract pixel data
image = ds.pixel_array

# Step 3: Display the X-ray
plt.figure(figsize=(7, 7))
plt.imshow(image, cmap="gray")
plt.title("First Chest X-ray")
plt.axis("off")
plt.show()
image = ds.pixel_array
print("Image shape:", image.shape)
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())
print("Average pixel value:", image.mean())
