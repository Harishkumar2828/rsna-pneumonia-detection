
import torch

from pathlib import Path
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "fasterrcnn_mobilenetv3_trained.pth"
)

DEVICE = torch.device("cpu")

NUM_CLASSES = 2
CONFIDENCE_THRESHOLD = 0.5


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {MODEL_PATH}"
        )

    print("Loading trained pneumonia detection model...")

    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None
    )

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Device:", DEVICE)

    return model
