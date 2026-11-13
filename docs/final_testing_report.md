\# Final Testing Report – RSNA Pneumonia Detection



\## 1. Testing Overview



This report summarizes the model evaluation, backend testing, preprocessing validation, inference testing and frontend integration testing performed for the RSNA Pneumonia Detection project.



\*\*Model:\*\* Faster R-CNN MobileNetV3 Large 320 FPN  

\*\*Final checkpoint:\*\* Epoch 3  

\*\*Execution device:\*\* CPU  

\*\*Input resolution:\*\* 512 × 512 × 3



\## 2. Model Evaluation Results



The model was evaluated on 5,337 validation images.



\### Epoch 1 vs Epoch 3



| Metric | Epoch 1 | Epoch 3 |

|---|---:|---:|

| True positives | 120 | 398 |

| False positives | 177 | 553 |

| False negatives | 1,766 | 1,488 |

| Precision | 40.40% | 41.85% |

| Recall | 6.36% | 21.10% |

| AP@0.50 | 13.09% | 20.74% |

| mAP@0.50:0.95 | 3.37% | 5.92% |

| Mean matched IoU | 0.6631 | 0.6715 |

| Average inference time/image | 0.132 sec | 0.109 sec |



\### Observations



\- Recall improved from 6.36% to 21.10%.

\- True positives increased from 120 to 398.

\- False negatives decreased from 1,766 to 1,488.

\- AP@0.50 improved from 13.09% to 20.74%.

\- mAP@0.50:0.95 improved from 3.37% to 5.92%.

\- Precision improved slightly from 40.40% to 41.85%.



The results indicate improvement between Epoch 1 and Epoch 3. However, detection performance remains limited, particularly in recall and mAP.



\## 3. Backend Testing



| Test | Expected Result | Status |

|---|---|---|

| Model loading | Load trained Epoch 3 checkpoint | PASS |

| Checkpoint verification | Report epoch 3 | PASS |

| Device verification | CPU execution | PASS |

| DICOM file reading | Read original DICOM | PASS |

| MONOCHROME handling | Apply appropriate inversion | Implemented |

| Image normalization | Pixel values between 0 and 1 | PASS |

| Image resizing | 512 × 512 | PASS |

| Tensor conversion | Shape (3, 512, 512) | PASS |

| Prediction execution | Generate detections | PASS |



\## 4. Inference Testing



An actual validation DICOM image was processed through the model.



Observed output:



\- Pneumonia detected: True

\- Detection count: 2

\- First detection confidence: 70.24%

\- Second detection confidence: 54.73%

\- Output bounding-box coordinates generated successfully.



The inference pipeline completed without errors.



\## 5. Flask API Testing



| Test | Result | Status |

|---|---|---|

| Flask application startup | Server started successfully | PASS |

| Model initialization | Epoch 3 model loaded | PASS |

| GET /api/health | Healthy response returned | PASS |

| POST /api/predict | Prediction response returned | PASS |

| JSON response | Detection information displayed | PASS |



\## 6. Frontend Integration Testing



A DICOM image was uploaded through the browser interface.



Observed results:



\- Upload completed successfully.

\- Image dimensions: 1024 × 1024.

\- Pneumonia opacity detection returned.

\- Detection count: 1.

\- Confidence: 59.73%.

\- Bounding-box coordinates displayed.

\- Model architecture and checkpoint information displayed.



\*\*Frontend integration test: PASSED\*\*



\## 7. Qualitative Evaluation



Prediction visualizations were generated for:



1\. True-positive example.

2\. False-positive prediction on a positive image.

3\. False-negative example.

4\. False-positive prediction on a negative image.

5\. Clean negative example.



The examples demonstrate that the model can identify some opacity regions but also produces missed detections and false positives.



Detailed observations are documented in `qualitative\_prediction\_analysis.md`.



\## 8. Overall Testing Summary



The following major components were successfully tested:



\- Dataset preprocessing pipeline.

\- Model checkpoint loading.

\- DICOM image preprocessing.

\- Model inference.

\- Prediction response generation.

\- Flask backend.

\- API health endpoint.

\- Frontend DICOM upload.

\- Frontend prediction display.



\*\*Overall application integration status: PASSED\*\*



\## 9. Known Limitations



\- Training was performed using a limited subset of 2,000 images.

\- Only three training epochs were completed.

\- Validation recall and mAP remain low.

\- False-positive and false-negative detections remain.

\- The application has been tested locally, not in a production environment.

\- Clinical validation has not been performed.



\## 10. Conclusion



The RSNA Pneumonia Detection project successfully demonstrates an end-to-end deep learning object detection workflow for chest X-ray DICOM images.



The trained Faster R-CNN model was integrated with a Flask backend and browser-based frontend. Model evaluation, inference, API functionality and frontend integration tests were completed successfully.



Although the application is technically functional, its detection performance is not sufficient for clinical use. Further training, optimization and independent validation would be required for future research.

