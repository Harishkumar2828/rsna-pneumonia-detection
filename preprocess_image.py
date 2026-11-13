
import pydicom
import cv2

# Step 1: Read the DICOM image
path = "data/processed/train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm"

ds = pydicom.dcmread(path)

# Step 2: Extract pixel data
image = ds.pixel_array

print("Original image shape:", image.shape)

# Step 3: Resize image
resized_image = cv2.resize(image, (224, 224))

print("Resized image shape:", resized_image.shape)

# Step 4: Check pixel values
print("Minimum pixel value:", resized_image.min())
print("Maximum pixel value:", resized_image.max())
# Step 5: Normalize pixel values
normalized_image = resized_image / 255.0

print("Normalized minimum:", normalized_image.min())
print("Normalized maximum:", normalized_image.max())
print("Normalized data type:", normalized_image.dtype)
# Step 6: Convert to float32
normalized_image = normalized_image.astype("float32")

print("Final data type:", normalized_image.dtype)
print("Final image shape:", normalized_image.shape)
# Step 7: Convert grayscale to 3 channels
image_3channel = cv2.cvtColor(
    normalized_image,
    cv2.COLOR_GRAY2RGB
)

print("Three-channel image shape:", image_3channel.shape)
print("Three-channel data type:", image_3channel.dtype)
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))
plt.imshow(image_3channel)
plt.title("Preprocessed Chest X-ray")
plt.axis("off")
plt.show()