"""
Configuration module for the Food Image Classification Web Application.
Contains all application-level settings including Flask, Redis, and model configurations.
"""

import os


class Config:
    """
    Base configuration class for the Flask application.
    Centralizes all configuration variables for easy management.
    """

    # ─── Flask Settings ─────────────────────────────────────────────────
    SECRET_KEY = os.environ.get('SECRET_KEY', 'food-classifier-secret-key-2026')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

    # ─── File Upload Settings ───────────────────────────────────────────
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

    # ─── Model Settings ────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    MODEL_PATHS = {
        'custom_cnn': os.path.join(BASE_DIR, 'Custom_CNN', 'custom_cnn_model.h5'),
        'vgg16': os.path.join(BASE_DIR, 'VGG16', 'vgg16_model.h5'),
        'resnet': os.path.join(BASE_DIR, 'ResNet', 'resnet_model.h5'),
    }

    MODEL_DISPLAY_NAMES = {
        'custom_cnn': 'Custom CNN',
        'vgg16': 'VGG16 (Transfer Learning)',
        'resnet': 'ResNet (Transfer Learning)',
    }

    METRICS_PATHS = {
        'custom_cnn': os.path.join(BASE_DIR, 'Custom_CNN', 'Custom_Model.txt'),
        'vgg16': os.path.join(BASE_DIR, 'VGG16', 'VGG16_Model.txt'),
        'resnet': os.path.join(BASE_DIR, 'ResNet', 'ResNet_Model.txt'),
    }

    TRAINING_PLOT_PATHS = {
        'custom_cnn': 'Custom_CNN/custom_cnn_training_plot.png',
        'vgg16': 'VGG16/vgg16_training_plot.png',
        'resnet': 'ResNet/resnet_training_plot.png',
    }

    # ─── Image Preprocessing ───────────────────────────────────────────
    IMAGE_SIZE = (256, 256)

    # ─── Nutrition Data ─────────────────────────────────────────────────
    NUTRITION_PATH = os.path.join(BASE_DIR, 'data', 'nutrition.json')

    # ─── Redis Configuration ───────────────────────────────────────────
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)
    REDIS_CACHE_TTL = int(os.environ.get('REDIS_CACHE_TTL', 3600))  # 1 hour

    # ─── Class Labels (sorted to match Keras model output training order) ───
    CLASS_LABELS = [
        'pakode', 'kulfi', 'Hot Dog', 'dhokla', 'masala_dosa',
        'chole_bhature', 'apple_pie', 'Crispy Chicken', 'burger',
        'chapati', 'paani_puri', 'sushi', 'dal_makhani', 'Donut',
        'Fries', 'cheesecake', 'omelette', 'Baked Potato', 'ice_cream',
        'pizza', 'Taquito', 'jalebi', 'chai', 'kaathi_rolls', 'Taco',
        'chicken_curry', 'pav_bhaji', 'butter_naan', 'momos', 'samosa',
        'fried_rice', 'Sandwich', 'idli', 'kadai_paneer'
    ]


class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production-specific configuration."""
    DEBUG = False


# ─── Config Selector ────────────────────────────────────────────────────
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
