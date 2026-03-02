# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. System dependencies for OpenCV/Pillow if needed
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gdown

# 5. Copy the entire project
COPY . .

# 6. Download the AI models from Google Drive during the build
RUN mkdir -p Custom_CNN VGG16 ResNet && \
    gdown --id 1bvpRGlc5F1Xriceb4_xZD9xcO52xR9QS -O Custom_CNN/custom_cnn_model.h5 && \
    gdown --id 1dvLCg6R1F7NJOvAW_6rxOBLZhGGG7Q57 -O VGG16/vgg16_model.h5 && \
    gdown --id 175WNbjCs_mZ7jImQaahRESRGaMigBl5r -O ResNet/resnet_model.h5

# 7. Expose the port
EXPOSE 5000

# 8. Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
