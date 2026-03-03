---
title: FoodVision AI
emoji: 🍕
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
pinned: true
---

<div align="center">
  
# 🧠 FoodVision AI: Deep Learning Diet & Nutrition Analyzer
  
**An end-to-end Computer Vision application that classifies 34 complex food categories using advanced Transfer Learning architectures (ResNet & VGG16) and serves real-time nutritional insights via a lightning-fast Flask & Redis backend.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Redis](https://img.shields.io/badge/Redis-5.0-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com)

</div>

<br>

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture & Tech Stack](#-system-architecture--tech-stack)
- [Deep Learning Models](#-deep-learning-models)
  - [Custom CNN](#1-custom-cnn-baseline)
  - [VGG16 Transfer Learning](#2-vgg16-transfer-learning)
  - [ResNet50 Transfer Learning](#3-resnet-transfer-learning-recommended)
- [Dataset Specifications](#-dataset-specifications)
- [Getting Started / Installation](#-getting-started)
- [Deployment Guide](#-deployment-guide)
- [Author](#-author)

---

## 🎯 Project Overview

**FoodVision AI** bridges the gap between state-of-the-art computer vision and everyday dietary tracking. By simply uploading an image of a meal, the system instantly identifies the food type and retrieves its complete macronutrient and micronutrient profile. 

Built with scalability and rapid inference in mind, the backend leverages **Redis caching** to ensure O(1) response times for identical or repeated image queries, bypassing the expensive neural network forward pass when possible. The frontend is crafted with a modern, responsive **Glassmorphism** UI, providing a premium user experience across all devices.

---

## ✨ Key Features

* 🔮 **Multi-Model Inference Engine**: Allows users to seamlessly toggle between three distinct convolutional neural network architectures (Custom CNN, VGG16, ResNet) to compare real-time inference results and confidence scores.
* ⚡ **Intelligent Redis Caching**: Hashes incoming image byte streams and caches the associated model predictions and nutritional lookups. Reduces repeated inference time from ~800ms to <10ms.
* 🍎 **Automated Nutritional Analysis**: Maps predicted classes against a comprehensive JSON database to return granular dietary insights (Calories, Proteins, Carbs, Fats, Fiber, Vitamins, Minerals).
* 📊 **Live Model Metrics**: Dynamically displays detailed validation metrics (Precision, Recall, F1-Score, Confusion Matrix metrics) for the selected model alongside the results.
* 🎨 **Premium Glassmorphism UI**: Features a sleek, modern, and fully responsive frontend with an integrated Dark/Light mode toggle, animated transitions, and drag-and-drop file upload capabilities.

---

## 💻 System Architecture & Tech Stack

### AI & Machine Learning Pipeline
* **TensorFlow / Keras (v2.16)**: Core framework for defining, compiling, and training the neural networks.
* **NumPy / Pandas**: Used extensively for dataset preprocessing, augmentation, and numerical manipulation.
* **Pillow (PIL)**: Backend image parsing, resizing to network input dimensions $224 \times 224 \times 3$, and channel normalization.

### Backend App Server
* **Python (v3.10+)**: Core runtime environment.
* **Flask (v3.1.0)**: Lightweight WSGI web application framework serving RESTful prediction endpoints and rendering Jinja2 templates.
* **Redis (v5.0)**: In-memory datastore utilized for the prediction caching layer.
* **Gunicorn**: High-performance Python WSGI HTTP Server for UNIX environments.

### Frontend Client
* **Vanilla JavaScript (ES6)**: Handles asynchronous model requests, dynamic DOM updates, and client-side validation without heavy framework overhead.
* **Bootstrap 5.3 & Custom CSS3**: Utilizes grid systems integrated with bespoke CSS variables, glassmorphism UI tokens, and keyframe animations.

---

## 🧠 Deep Learning Models

To demonstrate the efficacy of Transfer Learning in complex computer vision tasks, three distinct models were trained and evaluated on the dataset:

### 1. Custom CNN (Baseline)
A structural baseline Convolutional Neural Network built from scratch. 
* **Architecture**: Sequential structure with interleaved $Conv2D$ (ReLU activation) and $MaxPooling2D$ layers, flattened into fully connected $Dense$ networks paired with heavy $Dropout$ layers to combat overfitting.
* **Performance**: Achieved **~24% Accuracy**, establishing the difficulty of the dataset and the necessity for deeper architectures.

### 2. VGG16 (Transfer Learning)
Utilizes the robust VGG16 architecture pre-trained on ImageNet.
* **Architecture**: The deep convolutional base of VGG16 was frozen to retain foundational feature extraction (edges, colors, textures). A custom classification head featuring GlobalAveragePooling2D and Dense layers was attached and trained specifically on our 34 food classes.
* **Performance**: Achieved **~51% Accuracy**, demonstrating a massive leap in spatial feature recognition over the baseline.

### 3. ResNet (Transfer Learning) — *Recommended*
Leverages Residual Networks to solve the vanishing gradient problem inherent in ultra-deep networks.
* **Architecture**: Employs identity shortcut connections that skip one or more layers. Like VGG16, it utilizes ImageNet weights with a fine-tuned top dense classification head.
* **Performance**: **Best Overall Performance**. It consistently delivers the highest top-1 accuracy, greatest precision/recall stability, and superior robustness against complex, multi-item food imagery.

---

## 📦 Dataset Specifications

The models were trained on a specialized, heavily augmented dataset featuring complex, real-world lighting conditions and varying plating presentations.

* **Total Classes Supported (34)**: *Apple Pie, Baked Potato, Burger, Butter Naan, Chai, Chapati, Cheesecake, Chicken Curry, Chole Bhature, Crispy Chicken, Dal Makhani, Dhokla, Donut, Fried Rice, Fries, Hot Dog, Ice Cream, Idli, Jalebi, Kaathi Rolls, Kadai Paneer, Kulfi, Masala Dosa, Momos, Omelette, Paani Puri, Pakode, Pav Bhaji, Pizza, Samosa, Sandwich, Sushi, Taco, Taquito.*
* **Input Resolution**: $224 \times 224$ pixels (RGB).
* **Augmentation Strategy**: ImageGenerator was utilized to artificially expand the training distribution via random rotational shifts, horizontal/vertical flipping, dynamic zooming, and width/height shifting.

--- 

## 🚀 Getting Started

Follow these instructions to run the application locally on your machine.

### Prerequisites
* Python 3.10 or higher.
* Git.
* [Redis](https://redis.io/download/) (Optional, but highly recommended. The application features a graceful fallback if Redis is unreachable).

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/karthik-vana/FoodVision-AI.git
   cd FoodVision-AI
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Redis (Optional)**
   ```bash
   # Run this in a separate terminal window
   redis-server
   ```

5. **Launch the Application Server**
   ```bash
   python app.py
   ```
   *The application will be securely served on `http://127.0.0.1:5000`*

---

## 🌐 Deployment Guide 

This application is deployed to **Render** using its native **Python environment** (no Docker required).

The included `render.yaml` blueprint auto-configures the service:
- **Build**: `pip install -r requirements.txt`
- **Start**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
- **Models**: Automatically downloaded from Google Drive on first request via `gdown`.

**Note:** Vercel and similar serverless platforms enforce strict size limits that TensorFlow exceeds. Use PaaS solutions like **Render**, **Railway**, or **Heroku** which support larger runtimes and ephemeral file systems.

---

## 👤 Author

**Karthik Vana**
* **Role**: Data Engineer | AI Engineer
* **GitHub**: [@karthik-vana](https://github.com/karthik-vana)

---

<div align="center">
  <br>
  <i>Developed with precision as part of the Viharatech EdTech engineering internship submission.</i>
</div>
