\# System Architecture – RSNA Pneumonia Detection



\## 1. Architecture Overview



The project follows a modular architecture consisting of five main components:



1\. User Interface

2\. Flask Backend

3\. DICOM Preprocessing

4\. Deep Learning Inference

5\. Prediction Response



\## 2. System Workflow



```text

User

&#x20; |

&#x20; v

Frontend (HTML, CSS, JavaScript)

&#x20; |

&#x20; | Upload DICOM image

&#x20; v

Flask Backend (app.py)

&#x20; |

&#x20; v

DICOM Preprocessing

(image\_preprocessing.py)

&#x20; |

&#x20; | 512 x 512 x 3 tensor

&#x20; v

Model Loader

(model\_loader.py)

&#x20; |

&#x20; v

Faster R-CNN Model

(MobileNetV3 Large 320 FPN)

&#x20; |

&#x20; v

Prediction Module

(prediction.py)

&#x20; |

&#x20; | Detection scores and bounding boxes

&#x20; v

Flask JSON Response

&#x20; |

&#x20; v

Frontend displays results

```



\## 3. Component Description



\### 3.1 Frontend



File: `frontend/index.html`



Responsibilities:

\- Provides the image upload interface.

\- Accepts DICOM files.

\- Sends uploaded files to the Flask prediction endpoint.

\- Displays detection status, confidence scores and bounding-box coordinates.



\### 3.2 Flask Backend



File: `backend/app.py`



Responsibilities:

\- Initializes the Flask application.

\- Serves the frontend.

\- Handles API requests.

\- Validates uploaded files.

\- Coordinates preprocessing and prediction.

\- Returns JSON responses.



\### 3.3 DICOM Preprocessing



File: `backend/image\_preprocessing.py`



Responsibilities:

\- Reads DICOM pixel data using Pydicom.

\- Handles MONOCHROME1 image inversion.

\- Normalizes pixel values.

\- Resizes images to 512 × 512.

\- Converts grayscale images into three-channel PyTorch tensors.



\### 3.4 Model Loader



File: `backend/model\_loader.py`



Responsibilities:

\- Initializes the Faster R-CNN architecture.

\- Loads the trained Epoch 3 checkpoint.

\- Configures the detection confidence threshold.

\- Sets the model to evaluation mode.

\- Uses CPU for inference.



\### 3.5 Prediction Module



File: `backend/prediction.py`



Responsibilities:

\- Runs inference using the trained model.

\- Extracts predicted bounding boxes, labels and confidence scores.

\- Filters detections using the confidence threshold.

\- Returns structured prediction results.



\## 4. Model Architecture



\*\*Detection Framework:\*\* Faster R-CNN



\*\*Backbone:\*\* MobileNetV3 Large 320 FPN



\*\*Number of classes:\*\* 2



\- Class 0: Background

\- Class 1: Pneumonia Opacity



\*\*Input tensor:\*\* 3 × 512 × 512



\*\*Checkpoint:\*\* `fasterrcnn\_mobilenetv3\_trained.pth`



\*\*Training epochs:\*\* 3



\## 5. Data Flow



1\. The user selects a DICOM image.

2\. The frontend sends the file to `/api/predict`.

3\. Flask temporarily stores the uploaded file.

4\. The preprocessing module reads and transforms the image.

5\. The trained model generates predictions.

6\. Bounding-box coordinates are converted back to original image dimensions.

7\. Flask returns prediction details in JSON format.

8\. The frontend displays the results.

9\. The temporary uploaded file is deleted.



\## 6. Deployment Architecture



The current application runs locally.



\- Backend: Flask development server.

\- Frontend: HTML, CSS and JavaScript served by Flask.

\- Model execution: CPU.

\- Local address: `http://127.0.0.1:5000`



This is a local demonstration setup, not a production deployment.



\## 7. Security and Limitations



\- Uploaded files are restricted to `.dcm` extensions.

\- Maximum upload size is 20 MB.

\- Temporary uploaded files are deleted after processing.

\- The application is not clinically validated.

\- The model's prediction results should not be interpreted as medical diagnoses.



\## 8. Future Architecture Improvements



\- Production WSGI server.

\- GPU-based inference.

\- Improved input validation and error handling.

\- Authentication and access control.

\- Containerized deployment.

\- More robust monitoring and logging.

