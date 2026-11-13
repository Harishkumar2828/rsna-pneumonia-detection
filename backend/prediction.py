
import torch

from model_loader import DEVICE, CONFIDENCE_THRESHOLD


def predict_image(model, image):

    model.eval()

    image = image.to(DEVICE)

    with torch.no_grad():

        predictions = model([image])[0]

    boxes = predictions["boxes"].cpu()
    labels = predictions["labels"].cpu()
    scores = predictions["scores"].cpu()

    detections = []

    for box, label, score in zip(boxes, labels, scores):

        if int(label) != 1:
            continue

        if float(score) < CONFIDENCE_THRESHOLD:
            continue

        detections.append({
            "class_id": int(label),
            "class_name": "Pneumonia Opacity",
            "confidence": round(float(score), 4),
            "box_512": [
                round(float(value), 2)
                for value in box.tolist()
            ]
        })

    return {
        "pneumonia_detected": len(detections) > 0,
        "detection_count": len(detections),
        "detections": detections
    }
