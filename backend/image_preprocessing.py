
import numpy as np
import torch
import torch.nn.functional as F
import pydicom

from pathlib import Path


IMAGE_SIZE = 512


def preprocess_dicom(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"DICOM file not found: {image_path}"
        )

    # Read DICOM image
    dicom = pydicom.dcmread(str(image_path))

    image = dicom.pixel_array.astype(np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale DICOM image, got {image.shape}"
        )

    # Handle MONOCHROME1 images
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = image.max() + image.min() - image

    original_height, original_width = image.shape

    # Normalize between 0 and 1
    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum)
    else:
        image = np.zeros_like(image)

    # Convert to PyTorch tensor
    image = torch.from_numpy(image).float()

    # Add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)

    # Resize to 512 x 512
    image = F.interpolate(
        image,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    # Remove batch dimension
    image = image.squeeze(0)

    # Convert grayscale to three channels
    image = image.repeat(3, 1, 1)

    return image, original_height, original_width
