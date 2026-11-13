

\# Medical Imaging Project

\## Pneumonia Detection Using Faster R-CNN



\*\*Project:\*\* Project 1 – Medical Imaging  

\*\*Dataset:\*\* RSNA Pneumonia Detection Challenge  

\*\*Model:\*\* Faster R-CNN MobileNetV3 Large 320 FPN  

\*\*Training:\*\* 3 Epochs  

\*\*Status:\*\* Model Training and Evaluation Completed



\---



\## 1. Project Overview



The objective of this project is to develop a deep learning object detection model that identifies pneumonia regions in chest X-ray images.



The model predicts bounding boxes around suspected pneumonia regions.



Unlike image classification, which predicts a class for an entire image, object detection identifies both the presence and location of the target.



\## 2. Dataset Information



Dataset: RSNA Pneumonia Detection Challenge



Training dataset:

\- Total training images: 2,000

\- Positive images: 1,000

\- Negative images: 1,000



Validation dataset:

\- Total images: 5,337

\- Positive images: 1,203

\- Negative images: 4,134

\- Ground-truth bounding boxes: 1,886



The validation dataset was kept untouched during training and used for model evaluation.



\## 3. Model Architecture



Model: Faster R-CNN MobileNetV3 Large 320 FPN



Components:

\- Faster R-CNN: Object detection framework.

\- MobileNetV3 Large: Feature extraction backbone.

\- FPN: Feature Pyramid Network for multi-scale feature extraction.

\- RPN: Region Proposal Network.

\- ROI Head: Classification and bounding-box prediction.



Training configuration:

\- Optimizer: SGD

\- Learning rate: 0.001

\- Momentum: 0.9

\- Weight decay: 0.0005

\- Batch size: 2

\- Number of epochs: 3

\- Random seed: 42

\- Training device: CPU

\- Backbone: Frozen

\- RPN and ROI components: Trainable



Image preprocessing resolution: 512 x 512 pixels with 3 channels.



The 320 designation refers to the selected model architecture variant,

not the actual preprocessing resolution used in this training pipeline.



\## 4. Training Results



Training was performed in three epochs.



| Epoch | Average Training Loss |

|---|---:|

| 1 | 0.216784 |

| 2 | 0.270170 |

| 3 | 0.296392 |



Epoch 2 training time: 531.50 seconds  

Epoch 3 training time: 502.75 seconds



Resumed training time: 1,034.25 seconds.



Final trained checkpoint:



models/fasterrcnn\_mobilenetv3\_trained.pth



Training history:



models/training\_history.csv



\## 5. Evaluation Methodology



The model was evaluated using the untouched validation dataset.



Evaluation configuration:

\- Validation images: 5,337

\- Confidence threshold: 0.50

\- IoU threshold: 0.50

\- Evaluation metrics: Precision, Recall, Mean IoU, AP@0.50 and mAP@0.50:0.95.



Two evaluation reports were generated:



1\. evaluation/evaluation\_metrics.json

2\. evaluation/evaluation\_metrics\_epoch\_3.json



\## 6. Epoch 1 vs Epoch 3 Comparison



| Metric | Epoch 1 | Epoch 3 |

|---|---:|---:|

| Precision | 0.4040 | 0.4185 |

| Recall | 0.0636 | 0.2110 |

| Mean IoU | 0.6631 | 0.6715 |

| AP@0.50 | 0.1309 | 0.2074 |

| mAP@0.50:0.95 | 0.0337 | 0.0592 |



All five primary evaluation metrics improved after three epochs.



Recall showed the most significant improvement, increasing from approximately 6.36% to 21.10%.



This indicates that the trained model detected more actual pneumonia regions than the original Epoch 1 model.



\## 7. Detection Statistics



| Detection Metric | Epoch 1 | Epoch 3 |

|---|---:|---:|

| True Positives | 120 | 398 |

| False Positives | 177 | 553 |

| False Negatives | 1,766 | 1,488 |

| Predicted Boxes | 133,596 | 87,436 |



Observations:



\- True positives increased by 278.

\- False negatives decreased by 278.

\- False positives increased by 376.



The model improved its detection capability but still produces incorrect detections and misses many actual pneumonia regions.



\## 8. Average Precision at Different IoU Thresholds



| IoU Threshold | Epoch 1 AP | Epoch 3 AP |

|---|---:|---:|

| 0.50 | 0.1309 | 0.2074 |

| 0.55 | 0.0937 | 0.1563 |

| 0.60 | 0.0596 | 0.1101 |

| 0.65 | 0.0332 | 0.0672 |

| 0.70 | 0.0149 | 0.0355 |

| 0.75 | 0.0038 | 0.0115 |

| 0.80 | 0.0008 | 0.0031 |

| 0.85 | 0.0001 | 0.0007 |

| 0.90 | 0.000007 | 0.000025 |

| 0.95 | 0.0000 | 0.0000 |



The model performs better at lower IoU thresholds. Performance decreases significantly at stricter localization thresholds.



\## 9. Inference Performance



| Metric | Epoch 1 | Epoch 3 |

|---|---:|---:|

| Total Inference Time | 703.69 sec | 579.08 sec |

| Average Time per Image | 0.1319 sec | 0.1085 sec |



Epoch 3 achieved an observed inference time improvement during evaluation.



Average Epoch 3 inference time was approximately 0.109 seconds per image.



\## 10. Training Observations



Training loss increased across the three epochs:



\- Epoch 1: 0.216784

\- Epoch 2: 0.270170

\- Epoch 3: 0.296392



However, validation metrics improved.



This requires further investigation. Increasing training loss alone is insufficient to conclude that the model is overfitting.



Additional training experiments and loss-component analysis may help explain this behaviour.



\## 11. Model Limitations



Current limitations include:



1\. Low overall recall.

2\. High number of false-positive detections.

3\. Low mAP at stricter IoU thresholds.

4\. Increasing training loss across epochs.

5\. Limited training sample size compared with the full available dataset.

6\. Evaluation performance requires further improvement before any clinical application.



\## 12. Future Improvements



Potential improvements include:



\- Increasing the number of training samples.

\- Experimenting with additional training epochs.

\- Analysing classification and bounding-box regression losses separately.

\- Tuning the confidence threshold.

\- Investigating false-positive and false-negative predictions.

\- Experimenting with data augmentation.

\- Evaluating alternative detection architectures.

\- Comparing performance across different training configurations.



\## 13. Project Artifacts



Important generated files:



Final model:

models/fasterrcnn\_mobilenetv3\_trained.pth



Training history:

models/training\_history.csv



Epoch 1 evaluation:

evaluation/evaluation\_metrics.json



Epoch 3 evaluation:

evaluation/evaluation\_metrics\_epoch\_3.json



Epoch comparison:

evaluation/epoch\_1\_vs\_epoch\_3\_comparison.json



\## 14. Conclusion



The Faster R-CNN MobileNetV3 model demonstrated improved pneumonia detection performance after three training epochs.



The most significant improvement was observed in recall, which increased from 6.36% to 21.10%.



AP@0.50 improved from 13.09% to 20.74%, while mAP@0.50:0.95 improved from 3.37% to 5.92%.



Although the model showed measurable improvement, its detection performance still requires further optimization.



This project establishes a baseline for additional experimentation and improvement in pneumonia object detection.



\---



\*\*Project Status:\*\* Three-epoch training and evaluation completed.



\*\*Documentation Version:\*\* 1.0



