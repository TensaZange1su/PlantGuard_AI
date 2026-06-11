# PlantGuard AI

## AI-Based Smart Plant Monitoring and Disease Detection System

---

# Project Overview

PlantGuard AI is an intelligent plant monitoring system designed for automatic plant disease detection and plant condition analysis using deep learning, sensor simulation, and Large Language Models (LLMs).

The project combines:

* computer vision,
* environmental sensor analysis,
* neural network inference,
* explainable AI,
* and natural language diagnostics.

The system is intended for:

* smart home gardening,
* greenhouse monitoring,
* automated agricultural assistance,
* and AI-powered plant care support.

---

# Project Goals

The main goals of the project are:

1. Detect plant diseases from leaf images using deep learning.
2. Analyze environmental sensor parameters.
3. Generate AI-based explanations and recommendations.
4. Provide an interactive user interface through Streamlit.
5. Compare multiple neural network architectures for plant classification tasks.
6. Evaluate transfer learning and fine-tuning strategies for plant disease recognition.

---

# Datasets Used

## 1. PlantVillage Disease Dataset

The primary dataset used for large-scale plant disease classification experiments.

### Characteristics

* 38 plant disease classes
* Over 54,000 RGB leaf images
* Healthy and diseased plants
* Multi-class classification task

### Example Classes

* Apple Scab
* Tomato Early Blight
* Potato Late Blight
* Corn Rust
* Healthy plant classes

### Purpose

Used for:

* disease classification,
* transfer learning,
* fine-tuning,
* evaluation of deep learning architectures,
* large-scale benchmarking experiments.

The dataset was accessed using the HuggingFace Datasets library.

After validating the initial hypothesis using a lightweight Custom CNN pipeline, the project was later expanded to the larger PlantVillage dataset for advanced experiments using:

* ConvNeXtV2,
* EfficientNet-B3,
* transfer learning,
* and fine-tuning.

---

## 2. Plant Disease Detection Dataset

### Author: Karagwa Ann Treasure

A smaller plant disease dataset used during the early experimentation stage of the project.

### Characteristics

* smaller-scale dataset,
* fewer disease categories,
* lightweight training setup,
* suitable for rapid experimentation.

### Purpose

Used for:

* Custom CNN experiments,
* architecture prototyping,
* hypothesis validation,
* low-resource testing.

During the initial research phase, the hypothesis was tested using:

* a manually designed Custom CNN architecture,
* and a filtered subset containing 8 selected classes.

This stage was used to:

* validate the training pipeline,
* test CNN feasibility,
* compare lightweight architectures,
* and evaluate basic disease classification performance before scaling to larger datasets.

---

# Deep Learning Models

The project includes multiple neural network architectures for comparison.

---

# 1. ConvNeXt V2

## Architecture

ConvNeXt V2 is a modern convolutional neural network architecture inspired by Vision Transformers while maintaining CNN efficiency.

### Model Used

* ConvNeXtV2-Base

### Features

* transfer learning,
* fine-tuning,
* ImageNet pretrained weights,
* modern convolution blocks,
* high classification accuracy.

### Training Strategy

The model was trained using a two-phase transfer learning pipeline.

#### Phase 1 — Transfer Learning

* pretrained ImageNet weights,
* frozen backbone,
* training only classification head.

#### Phase 2 — Fine-Tuning

* backbone unfreezing,
* lower learning rate,
* adaptation to plant disease domain.

### Performance

The model achieved approximately:

* 97–98% validation accuracy,
* strong weighted F1-score,
* robust multiclass performance.

### Advantages

* strongest overall performance,
* robust feature extraction,
* excellent multiclass classification results.

### Limitations

* high VRAM usage,
* computationally expensive,
* slower training.

---

# 2. EfficientNet-B3

## Architecture

EfficientNet-B3 uses compound scaling to balance:

* depth,
* width,
* image resolution.

### Features

* transfer learning,
* fine-tuning,
* efficient parameter usage,
* lower computational cost,
* strong transfer learning performance.

### Training Pipeline

Implemented using:

* transfer learning phase,
* fine-tuning phase,
* evaluation dashboards,
* metric history logging,
* training analytics visualization.

### Performance

Achieved approximately:

* 95–96% validation accuracy,
* strong weighted F1-score,
* efficient inference performance.

### Advantages

* faster inference,
* lower memory usage,
* suitable for deployment,
* efficient training pipeline.

### Limitations

* slightly weaker than ConvNeXtV2 in difficult disease classes.

---

# 3. Custom CNN

## Architecture

A manually designed convolutional neural network.

### Structure

Includes:

* convolution layers,
* batch normalization,
* max pooling,
* ReLU activations,
* fully connected classifier.

### Dataset

Used with:

* the Plant Disease Detection dataset,
* and a filtered 8-class subset.

### Purpose

* architecture experimentation,
* lightweight benchmarking,
* educational implementation,
* hypothesis validation.

### Advantages

* simple architecture,
* fast training,
* minimal hardware requirements.

### Limitations

* lower accuracy,
* weaker generalization,
* limited scalability.

---

# Transfer Learning Pipeline

The project uses a two-phase transfer learning strategy.

## Phase 1 — Transfer Learning

* pretrained ImageNet weights,
* frozen feature extractor,
* training only classification head.

## Phase 2 — Fine-Tuning

* partial or full unfreezing,
* low learning rate,
* feature adaptation to plant diseases.

This approach improves:

* convergence speed,
* stability,
* validation accuracy,
* weighted F1-score,
* and overall model performance.

---

# Data Augmentation

Several augmentation techniques were applied during training:

* random resized crop,
* horizontal flipping,
* rotation,
* affine transformations,
* color jitter,
* normalization.

### Purpose

* improve generalization,
* reduce overfitting,
* simulate real-world variability.

---

# Sensor Layer

The project includes a simulated environmental sensor system.

## Simulated Parameters

* soil moisture,
* air humidity,
* temperature,
* light intensity,
* pH level.

### Example Usage

Sensor values are combined with image predictions to improve contextual analysis.

Example:

* low moisture + leaf damage
  → drought stress recommendation.

---

# LLM Layer

The project integrates a Large Language Model (LLM) explanation layer.

## Functions

* explain disease predictions,
* generate treatment recommendations,
* answer user questions,
* provide human-readable diagnostics,
* generate contextual plant-care suggestions.

### Example Outputs

* disease description,
* possible causes,
* treatment steps,
* watering recommendations,
* environmental suggestions.

---

# Explainable AI (XAI)

The system supports Grad-CAM visualization.

## Grad-CAM

Grad-CAM highlights image regions used by the neural network during prediction.

### Purpose

* interpretability,
* transparency,
* debugging,
* user trust.

Highlighted regions indicate areas that influenced model predictions.

---

# Streamlit Application

A Streamlit web application was developed for interaction with the models.

## Features

* image upload,
* disease prediction,
* top-3 predictions,
* confidence scores,
* Grad-CAM visualization,
* sensor simulation,
* AI-generated explanations.

---

# Evaluation Metrics

The models were evaluated using:

* accuracy,
* precision,
* recall,
* F1-score,
* confusion matrix,
* classification reports,
* per-class metrics heatmaps.

---

# Training Analytics

Training pipelines include:

* metric history logging,
* validation tracking,
* separate metric visualization,
* transfer learning vs fine-tuning comparison graphs.

Generated evaluation plots include:

* accuracy curves,
* loss curves,
* precision curves,
* recall curves,
* F1-score curves,
* confusion matrices,
* per-class metrics heatmaps,
* training phase comparison graphs.

---

# Technologies Used

## Deep Learning

* PyTorch
* timm
* torchvision

## Data Processing

* NumPy
* OpenCV
* HuggingFace Datasets

## Visualization

* Matplotlib
* Seaborn

## Explainability

* Grad-CAM

## Web Interface

* Streamlit

## AI Integration

* OpenAI API
* LLM-based explanation layer

---

# Hardware

Training was performed using:

* NVIDIA RTX 4050 Laptop GPU
* CUDA acceleration
* mixed precision training (FP16)

Additional cloud experiments were performed using:

* Google Colab Tesla T4 GPU.

---

# Future Improvements

Possible future extensions include:

* real IoT sensor integration,
* mobile deployment,
* real-world field dataset collection,
* multilingual AI assistant,
* edge AI optimization,
* greenhouse automation.

---

# Conclusion

PlantGuard AI demonstrates the integration of:

* computer vision,
* environmental sensing,
* explainable AI,
* transfer learning,
* and language models

for intelligent plant monitoring and disease analysis.

The project achieved strong classification performance across multiple architectures and provides a foundation for future smart agriculture systems.
