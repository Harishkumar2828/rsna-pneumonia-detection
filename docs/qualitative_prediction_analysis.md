\# Qualitative Prediction Analysis – RSNA Pneumonia Detection



\## 1. Overview



This report documents the qualitative evaluation of the Epoch 3 Faster R-CNN MobileNetV3 Large 320 FPN model.



Five visualization examples were generated from the validation dataset to inspect correct detections, missed detections, false alarms, and correct negative predictions.



The model uses 512 × 512 × 3 preprocessed chest X-ray images.



\## 2. Visualization Results



\### 2.1 True Positive and False Positive on a Positive Image



\*\*Patient ID:\*\* d4b2b93e-696e-47fa-b197-160a35534b84



\- Ground-truth boxes: 2

\- True Positives: 1

\- False Positives: 1

\- False Negatives: 1

\- True Positive confidence: 0.54

\- False Positive confidence: 0.58



\*\*Observations:\*\*



\- The model correctly detected one annotated pneumonia region.

\- One ground-truth region was missed.

\- One additional predicted box did not match a ground-truth box at the selected IoU threshold.

\- The predicted bounding boxes were not tightly aligned with the annotations.



The files `true\_positive\_example.png` and `false\_positive\_positive\_image.png` represent the same patient and prediction result.



\### 2.2 False Negative Example



\*\*Patient ID:\*\* cf8c70d8-1d3d-4cc6-b5bf-63713862da0c



\- Ground-truth boxes: 2

\- True Positives: 1

\- False Positives: 0

\- False Negatives: 1

\- True Positive confidence: 0.58



\*\*Observations:\*\*



\- The model detected one of the two annotated regions.

\- The second ground-truth region was missed.

\- The predicted box was not perfectly aligned with the corresponding ground-truth box.



This example demonstrates incomplete detection of annotated regions.



\### 2.3 False Positive on a Negative Image



\*\*Patient ID:\*\* e55c8aab-1e83-440c-80c7-ef01bd82c5f6



\- Ground-truth boxes: 0

\- True Positives: 0

\- False Positives: 2

\- False Negatives: 0

\- False Positive confidence scores: 0.70 and 0.55



\*\*Observations:\*\*



\- The image contains no annotated pneumonia bounding boxes.

\- The model generated two predictions.

\- Both predictions are false positives for this example.



This demonstrates that the model can generate incorrect detections on negative images.



\### 2.4 Clean Negative Example



\*\*Patient ID:\*\* ad7dbcfa-406f-4d54-8d85-0fe482ba4ca3



\- Ground-truth boxes: 0

\- True Positives: 0

\- False Positives: 0

\- False Negatives: 0



\*\*Observations:\*\*



\- The model generated no bounding-box predictions.

\- This is the desired detection behaviour for this negative example.



\## 3. Overall Qualitative Findings



The visualization analysis revealed the following:



1\. The model can identify some annotated pneumonia regions.

2\. Some ground-truth regions remain undetected.

3\. False-positive predictions occur on both positive and negative examples.

4\. Some predicted bounding boxes have imperfect localization.

5\. The model can correctly produce no detections for certain negative images.



\## 4. Connection with Quantitative Evaluation



The qualitative observations are consistent with the Epoch 3 validation metrics:



| Metric | Epoch 3 |

|---|---:|

| Precision | 41.85% |

| Recall | 21.10% |

| AP@0.50 | 20.74% |

| mAP@0.50:0.95 | 5.92% |



The relatively low recall indicates that many annotated regions are missed. The precision result also indicates that a considerable number of predictions are incorrect.



The visualizations provide examples of these limitations.



\## 5. Limitations



\- Only selected qualitative examples are shown in this report.

\- These examples are not a replacement for full validation metrics.

\- The model's predictions are not medical diagnoses.

\- Further model development and evaluation are required before any clinical use could be considered.



\## 6. Conclusion



The Epoch 3 model demonstrates the ability to detect some pneumonia regions, but its performance is limited by missed detections, false positives, and imperfect bounding-box localization.



The qualitative analysis complements the quantitative evaluation and provides visual evidence of the model's current strengths and weaknesses.



\*\*Step 16.11 – Qualitative Prediction Analysis: Completed.\*\*

