
import tempfile

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from model_loader import load_model
from image_preprocessing import preprocess_dicom
from prediction import predict_image


PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Load model once when the application starts
model = load_model()


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model": "Faster R-CNN MobileNetV3 Large 320 FPN",
        "checkpoint": "Epoch 3",
        "device": "CPU"
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No DICOM file uploaded."
        }), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({
            "error": "Please select a file."
        }), 400

    if not uploaded_file.filename.lower().endswith(".dcm"):
        return jsonify({
            "error": "Only .dcm files are supported."
        }), 400

    temporary_path = None

    try:

        # Save uploaded DICOM temporarily
        with tempfile.NamedTemporaryFile(
            suffix=".dcm",
            delete=False
        ) as temporary_file:

            uploaded_file.save(temporary_file)
            temporary_path = Path(temporary_file.name)

        # Preprocess image
        image, original_height, original_width = (
            preprocess_dicom(temporary_path)
        )

        # Run model prediction
        result = predict_image(model, image)

        # Convert bounding boxes to original image dimensions
        scale_x = original_width / 512
        scale_y = original_height / 512

        for detection in result["detections"]:

            x1, y1, x2, y2 = detection.pop("box_512")

            detection["bounding_box"] = {
                "x1": round(x1 * scale_x, 2),
                "y1": round(y1 * scale_y, 2),
                "x2": round(x2 * scale_x, 2),
                "y2": round(y2 * scale_y, 2)
            }

        result["image_dimensions"] = {
            "width": original_width,
            "height": original_height
        }

        result["model"] = "Faster R-CNN MobileNetV3 Large 320 FPN"
        result["checkpoint_epoch"] = 3

        return jsonify(result), 200

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 400

    finally:

        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "File too large. Maximum size is 20 MB."
    }), 413


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
