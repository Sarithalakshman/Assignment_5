# Assignment_5
Multiclass Fish Image Classification
1.Project Overview

This project aims to classify images of fish into multiple categories using Deep Learning and Transfer Learning techniques. Several pre-trained CNN architectures were evaluated to determine the best-performing model for deployment.
The final model — MobileNetV2 — achieved an outstanding 98.20% accuracy on the test dataset, demonstrating excellent generalization and efficiency.

2.Objectives

Classify fish images into multiple species categories.
Compare performance across different deep learning architectures.
Develop a Streamlit web app for real-time fish classification.
Provide complete documentation, metrics, and visualization.

3.Skills Learned / Tools Used

Deep Learning
Python, TensorFlow, Keras
Transfer Learning
Model Evaluation and Visualization

4.Dataset

The dataset contains multiple classes of fish images organized into folders by species.
Images were loaded using TensorFlow’s ImageDataGenerator for efficient preprocessing and augmentation.
Data augmentation techniques such as rotation, flipping, and zoom were applied to improve model robustness.

5.Approach
🔹 Data Preprocessing
Normalized images to the [0, 1] range.
Applied data augmentation to improve generalization.
🔹 Model Training
Trained one CNN model from scratch.
6.Fine-tuned five pre-trained models:

⚙️VGG16
⚙️ResNet50
⚙️MobileNetV2
⚙️InceptionV3
⚙️EfficientNetB0

🔹 Model Evaluation
Compared models based on Accuracy and Loss.
Visualized training and validation metrics.
Selected the best model (MobileNetV2) for deployment.

📊 Model Evaluation Results
Model Name	Test Accuracy	Test Loss
MobileNetV2	0.9820	0.0674
ResNet50	0.2595	2.0710
InceptionV3	0.9732	0.0957
EfficientNetB0	0.1641	2.3045
VGG16	0.7772	0.9956

✅ Best Performing Model: MobileNetV2
