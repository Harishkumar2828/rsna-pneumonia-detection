\# API Documentation – Pneumonia Detection System



\## 1. Introduction



This document describes the REST API provided by the RSNA Pneumonia Detection application.



The API uses Flask and provides endpoints for checking application health and performing pneumonia opacity detection on uploaded DICOM images.



\## 2. Base URL



Local development URL:



`http://127.0.0.1:5000`



\## 3. Health Check Endpoint



\### Endpoint



`GET /api/health`



\### Purpose



Checks whether the Flask application and model are available.



\### Example Request



```bash

curl http://127.0.0.1:5000/api/health

```



\### Example Response



```json

{

&#x20; "status": "healthy",

&#x20; "model": "Faster R-CNN MobileNetV3 Large 320 FPN",

&#x20; "checkpoint": "Epoch 3",

&#x20; "device": "CPU"

}

```



\### Success Status



`200 OK`



\## 4. Pneumonia Prediction Endpoint



\### Endpoint



`POST /api/predict`



\### Purpose



Accepts a DICOM image and returns pneumonia opacity detection results.



\### Request Format



\- Content-Type: `multipart/form-data`

\- Field name: `file`

\- Supported file extension: `.dcm`

\- Maximum file size: 20 MB



\### Example Request



```bash

curl -X POST http://127.0.0.1:5000/api/predict ^

&#x20; -F "file=@sample.dcm"

```



\### Example Successful Response



```json

{

&#x20; "pneumonia\_detected": true,

&#x20; "detection\_count": 1,

&#x20; "detections": \[

&#x20;   {

&#x20;     "class\_id": 1,

&#x20;     "class\_name": "Pneumonia Opacity",

&#x20;     "confidence": 0.5973,

&#x20;     "bounding\_box": {

&#x20;       "x1": 269.32,

&#x20;       "y1": 249.06,

&#x20;       "x2": 522.12,

&#x20;       "y2": 739.54

&#x20;     }

&#x20;   }

&#x20; ],

&#x20; "image\_dimensions": {

&#x20;   "width": 1024,

&#x20;   "height": 1024

&#x20; },

&#x20; "model": "Faster R-CNN MobileNetV3 Large 320 FPN",

&#x20; "checkpoint\_epoch": 3

}

```



This is an illustrative example based on a successful application test. Actual predictions will vary depending on the input image.



\### Success Status



`200 OK`



\## 5. Error Responses



| Status | Condition |

|---|---|

| 400 | Missing file, empty filename, unsupported extension or processing error |

| 413 | Uploaded file exceeds 20 MB |



\### Example Error Response



```json

{

&#x20; "error": "Only .dcm files are supported."

}

```



\## 6. Prediction Response Fields



| Field | Description |

|---|---|

| pneumonia\_detected | Whether any pneumonia opacity detection was returned |

| detection\_count | Number of returned detections |

| class\_id | Predicted class identifier |

| class\_name | Predicted class name |

| confidence | Model confidence score |

| bounding\_box | Coordinates in original image dimensions |

| image\_dimensions | Original image width and height |

| model | Model architecture |

| checkpoint\_epoch | Trained checkpoint epoch |



\## 7. Detection Threshold



The application uses a confidence threshold of `0.5`.



Predictions below this threshold are excluded from the returned detections.



\## 8. Technical Notes



\- Input images are processed as 512 × 512 three-channel tensors.

\- Original image dimensions are retained.

\- Predicted bounding boxes are scaled back to original image dimensions.

\- Inference is performed using CPU.

\- The model is loaded once when the Flask application starts.



\## 9. Important Disclaimer



The API is designed for educational and research demonstrations only. It is not clinically validated and must not be used for medical diagnosis or treatment decisions.

