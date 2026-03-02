# 🧠 AI Food Image Classification System

> **Classify food images into 34 categories with nutritional insights using Deep Learning & Transfer Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Redis](https://img.shields.io/badge/Redis-5.0-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Dataset Description](#-dataset-description)
- [Validation Results](#-validation-results)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [Usage](#-usage)
- [Deployment Guide](#-deployment-guide)
- [Screenshots](#-screenshots)
- [Author](#-author)

---

## 🎯 Project Overview

This is a **production-ready Food Image Classification Web Application** that uses three deep learning models to classify food images into **34 categories** and provide detailed nutritional information.

The system features:
- **3 AI Models**: Custom CNN, VGG16 (Transfer Learning), ResNet (Transfer Learning)
- **Real-time Prediction**: Upload a food image and get instant classification
- **Nutritional Insights**: Calories, protein, carbs, fat, fiber, vitamins, minerals
- **Redis Caching**: Faster repeated predictions with intelligent caching
- **Premium UI**: Glassmorphism design with dark/light mode toggle

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Multi-Model Prediction** | Choose between Custom CNN, VGG16, or ResNet |
| 📊 **Model Metrics** | View accuracy, precision, recall, F1-score, TP/TN/FP/FN |
| 🍎 **Nutrition Info** | Detailed nutritional breakdown per food item |
| ⚡ **Redis Caching** | Cached predictions for instant repeated lookups |
| 🎨 **Premium UI** | Glassmorphism cards, animations, responsive design |
| 🌙 **Dark/Light Mode** | Toggle between themes with persistent preference |
| 📱 **Fully Responsive** | Works on desktop, tablet, and mobile |
| 🖱️ **Drag & Drop** | Intuitive image upload with drag-and-drop |
| 🔔 **Toast Notifications** | Real-time feedback for user actions |

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10+** — Core language
- **Flask 3.1** — Web framework
- **TensorFlow / Keras 2.16** — Deep learning framework
- **Redis 5.0** — In-memory caching
- **NumPy** — Numerical computing
- **Pillow** — Image processing

### Frontend
- **HTML5** — Semantic structure
- **CSS3** — Custom styling with glassmorphism
- **JavaScript (ES6+)** — Interactive functionality
- **Bootstrap 5.3** — Responsive grid and components
- **Bootstrap Icons** — Icon library

### DevOps
- **Vercel** — Deployment platform
- **Git / GitHub** — Version control
- **Gunicorn** — WSGI HTTP server

---

## 🏗️ Model Architecture

### 1. Custom CNN
A custom-built convolutional neural network designed from scratch:
- Multiple Conv2D layers with ReLU activation
- MaxPooling for spatial dimensionality reduction
- Dense layers with Dropout for regularization
- Softmax output layer (34 classes)

### 2. VGG16 (Transfer Learning)
Pre-trained VGG16 model from ImageNet:
- Frozen base layers for feature extraction
- Custom dense head for food classification
- Fine-tuned on food dataset
- Improved accuracy over Custom CNN

### 3. ResNet (Transfer Learning)
Pre-trained ResNet model from ImageNet:
- Residual connections for deep feature learning
- Custom classification head
- Fine-tuned for food image domain
- Best performance among all models

---

## 📦 Dataset Description

- **Total Classes**: 34 food categories
- **Categories include**: Apple Pie, Baked Potato, Burger, Butter Naan, Chai, Chapati, Cheesecake, Chicken Curry, Chole Bhature, Crispy Chicken, Dal Makhani, Dhokla, Donut, Fried Rice, Fries, Hot Dog, Ice Cream, Idli, Jalebi, Kaathi Rolls, Kadai Paneer, Kulfi, Masala Dosa, Momos, Omelette, Paani Puri, Pakode, Pav Bhaji, Pizza, Samosa, Sandwich, Sushi, Taco, Taquito
- **Split**: Training / Validation / Testing
- **Image Size**: 224 × 224 pixels (RGB)
- **Augmentation**: Random rotation, flip, zoom, shift

---

## 📊 Validation Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | 24.24% | 20.70% | 24.24% | 20.77% |
| VGG16 | 51.52% | 56.75% | 51.52% | 51.13% |
| ResNet | Best | Best | Best | Best |

> Note: VGG16 and ResNet show significant improvements over Custom CNN due to transfer learning from ImageNet features.

---

## 📁 Project Structure

```
Task-2/
│
├── app.py                          # Flask application (routes + factory)
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel deployment config
├── .gitignore                      # Git exclusions
├── README.md                       # This file
│
├── utils/                          # Utility classes (OOP)
│   ├── __init__.py
│   ├── model_loader.py             # ModelLoader class
│   ├── image_preprocessor.py       # ImagePreprocessor class
│   ├── predictor.py                # Predictor class
│   ├── metrics_reader.py           # MetricsReader class
│   ├── nutrition_loader.py         # NutritionLoader class
│   └── redis_cache.py              # RedisCache class
│
├── data/
│   └── nutrition.json              # Nutritional data for 34 classes
│
├── Custom_CNN/
│   ├── custom_cnn_model.h5         # Trained model
│   ├── Custom_Model.txt            # Validation report
│   └── custom_cnn_training_plot.png
│
├── VGG16/
│   ├── vgg16_model.h5              # Trained model
│   ├── VGG16_Model.txt             # Validation report
│   └── vgg16_training_plot.png
│
├── ResNet/
│   ├── resnet_model.h5             # Trained model
│   ├── ResNet_Model.txt            # Validation report
│   └── resnet_training_plot.png
│
├── templates/                      # Jinja2 HTML templates
│   ├── landing.html
│   ├── index.html
│   ├── result.html
│   └── about.html
│
├── static/
│   ├── css/
│   │   └── style.css               # Custom CSS design system
│   ├── js/
│   │   └── main.js                 # Client-side JavaScript
│   ├── images/
│   └── uploads/                    # Uploaded images (gitignored)
│
└── image_Dataset/                  # Training dataset (gitignored)
    ├── training_data/
    ├── valid_data/
    └── testing_data/
```

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.10+
- pip
- Redis Server (optional, graceful fallback)
- Git

### Step-by-step Setup

```bash
# 1. Clone the repository
git clone https://github.com/karthik-vana/food-image-classification.git
cd food-image-classification

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Start Redis server
redis-server

# 6. Run the application
python app.py
```

The app will be available at `http://localhost:5000`

---

## 📖 Usage

1. **Open the app** → You'll see the animated landing page
2. **Click "Click to Enter"** → Navigate to the prediction page
3. **Upload a food image** → Drag & drop or browse files
4. **Select a model** → Custom CNN, VGG16, or ResNet
5. **Click "Classify Food"** → Wait for prediction
6. **View results** → Predicted class, confidence, nutrition, metrics

---

## 🌐 Deployment Guide (Vercel)

### Steps:
1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com) and log in
3. Click "New Project" → Import your GitHub repository
4. Configure environment variables:
   - `REDIS_HOST` = your Redis cloud host
   - `REDIS_PORT` = 6379
   - `REDIS_PASSWORD` = your Redis password
5. Click "Deploy"

### Important Notes:
- Model `.h5` files must be hosted externally (e.g., Google Drive, S3) due to Vercel's file size limits
- Use Redis Cloud (e.g., Redis Labs) for production caching
- Set `FLASK_ENV=production` in environment variables

---

## 📸 Screenshots

> Screenshots will be added after deployment.

---

## 👤 Author

**Karthik Vana**
- Role: Data Engineer | AI Engineer
- GitHub: [@karthik-vana](https://github.com/karthik-vana)

---

## 📝 License

This project is developed as part of the Viharatech EdTech internship submission.

---

<p align="center">
  <strong>🧠 Powered by Deep Learning | Built with ❤️ by Karthik Vana</strong>
</p>
