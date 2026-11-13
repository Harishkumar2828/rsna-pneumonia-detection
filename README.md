\# RSNA Pneumonia Detection Using Deep Learning



\## 1. Project Overview



This project develops a deep learning-based medical imaging system to detect pneumonia-related lung opacities in chest X-ray DICOM images.



The system uses a Faster R-CNN object detection model with a MobileNetV3 Large 320 FPN backbone. It identifies suspected opacity regions and returns detection confidence scores and bounding-box coordinates.



A Flask backend and web-based frontend provide an interface for uploading DICOM images and viewing prediction results.



\*\*This project is an educational research prototype and is not intended for clinical diagnosis or medical decision-making.\*\*



\## 2. Technologies Used



\- Python 3.11

\- PyTorch

\- Torchvision

\- Faster R-CNN

\- MobileNetV3 Large 320 FPN

\- NumPy and Pandas

\- Pydicom

\- Matplotlib

\- Flask

\- HTML, CSS and JavaScript



\## 3. Dataset



The project uses the RSNA Pneumonia Detection Challenge dataset.



\- Training subset used: 2,000 images

\- Validation images: 5,337

\- Positive validation images: 1,203

\- Negative validation images: 4,134

\- Ground-truth bounding boxes in validation data: 1,886



The validation dataset was kept separate from model training.



The original dataset is not included in this repository. Users must obtain the dataset separately and configure the expected project data paths.



\## 4. Image Preprocessing



The preprocessing pipeline includes:



1\. Reading DICOM images using Pydicom.

2\. Extracting pixel data.

3\. Inverting MONOCHROME1 images where necessary.

4\. Applying min-max normalization.

5\. Resizing images to 512 × 512 pixels.

6\. Converting grayscale images into three-channel tensors.

7\. Preparing bounding-box coordinates for model input.



The model receives images with a tensor shape of `(3, 512, 512)`.



\## 5. Model Architecture



\*\*Object Detection Model:\*\* Faster R-CNN



\*\*Backbone:\*\* MobileNetV3 Large 320 FPN



\*\*Classes:\*\*

\- Background

\- Pneumonia Opacity



The model was trained for three epochs using a CPU-based training environment.



The final trained checkpoint is stored locally at:



`models/fasterrcnn\_mobilenetv3\_trained.pth`



\## 6. Model Evaluation Results



Evaluation results from the Epoch 3 checkpoint:



| Metric | Result |

|---|---:|

| Validation images | 5,337 |

| True positives | 398 |

| False positives | 553 |

| False negatives | 1,488 |

| Precision | 41.85% |

| Recall | 21.10% |

| AP@0.50 | 20.74% |

| mAP@0.50:0.95 | 5.92% |

| Mean matched IoU | 0.6715 |

| Average inference time | 0.109 seconds/image |



These results indicate that the model can identify some pneumonia opacity regions, but its detection performance remains limited. Further training, tuning and validation are required before any practical clinical application could be considered.



\## 7. Application Features



\- Upload chest X-ray DICOM files.

\- Preprocess uploaded images automatically.

\- Run pneumonia opacity detection.

\- Display detection status.

\- Display the number of detected regions.

\- Display confidence scores.

\- Return bounding-box coordinates.

\- Display image dimensions and model checkpoint information.



\## 8. Project Structure



```text

Project\_1\_Medical\_Imaging/

│

├── backend/

│   ├── app.py

│   ├── model\_loader.py

│   ├── image\_preprocessing.py

│   └── prediction.py

│

├── frontend/

│   └── index.html

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   └── fasterrcnn\_mobilenetv3\_trained.pth

│

├── evaluation/

│   ├── evaluation\_metrics.json

│   ├── evaluation\_metrics\_epoch\_3.json

│   ├── epoch\_1\_vs\_epoch\_3\_comparison.json

│   └── visualizations/

│

├── docs/

│   ├── model\_training\_evaluation\_report.md

│   └── qualitative\_prediction\_analysis.md

│

├── notebooks/

├── src/

├── configs/

├── requirements.txt

└── README.md

```
## Model Checkpoint

The trained model checkpoint is not included in this repository because of its large file size.

**Expected checkpoint location:**
```text
models/fasterrcnn_mobilenetv3_trained.pth
```

To run the prediction application, obtain the trained checkpoint separately and place it at the path shown above.

If the checkpoint is unavailable, you can reproduce the training process using the training scripts provided in this repository, together with the RSNA Pneumonia Detection dataset.

**Note:** The dataset and trained model weights are not distributed with this repository. This project is an educational demonstration and is not intended for clinical diagnosis.


\## 9. Installation and Setup



\### Step 1: Clone the repository



```bash

git clone git clone https://github.com/Harishkumar2828/rsna-pneumonia-detection.git
cd Project\_1\_Medical\_Imaging

```



\### Step 2: Create a virtual environment



```bash

python -m venv .venv

```



Activate it on Windows:



```bash

.venv\\Scripts\\activate

```



\### Step 3: Install dependencies



```bash

pip install -r requirements.txt

```



Install compatible PyTorch and Torchvision builds for your hardware if they are not installed successfully through the requirements file.



\### Step 4: Configure project data and model



Place the trained checkpoint at:



`models/fasterrcnn\_mobilenetv3\_trained.pth`



Configure the original dataset and processed metadata in the expected project locations.



\### Step 5: Start the application



```bash

python backend/app.py

```



Open the application in your browser:



`http://127.0.0.1:5000`



\## 10. API Endpoints



\### Health Check



\*\*GET\*\* `/api/health`



Returns application and model status.



\### Pneumonia Prediction



\*\*POST\*\* `/api/predict`



Accepts a `.dcm` file using multipart form-data with the field name `file`.



Returns:

\- Pneumonia detection status

\- Detection count

\- Confidence scores

\- Bounding-box coordinates

\- Image dimensions

\- Model and checkpoint information



\## 11. Testing



The following application tests were completed successfully:



\- Model checkpoint loading.

\- DICOM image preprocessing.

\- Tensor shape and pixel normalization validation.

\- Model inference on a validation image.

\- Flask application startup.

\- API health endpoint.

\- Frontend DICOM upload and prediction display.



Qualitative prediction examples and evaluation results are available in the `evaluation/` and `docs/` folders.



\## 12. Limitations and Future Improvements



\- Training was performed on a limited subset of the dataset.

\- The model was trained for only three epochs.

\- Recall and mAP remain relatively low.

\- CPU training and inference limit experimentation speed.

\- False-positive and false-negative detections remain.

\- The model has not undergone clinical validation.



Future improvements include:

\- Training on a larger dataset.

\- Increasing the number of training epochs.

\- Hyperparameter tuning.

\- Confidence-threshold optimization.

\- Improved object detection evaluation.

\- GPU-based training.

\- Additional validation on independent datasets.



\## 13. Disclaimer



This application is developed strictly for educational and research purposes. It is not a certified medical device and must not be used to diagnose, treat or make decisions about patients.



\## 14. Author



Developed as a medical imaging and deep learning project using the RSNA Pneumonia Detection Challenge dataset.

