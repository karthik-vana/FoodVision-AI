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

# 5. Copy the entire project
COPY . .

# 6. Expose the port
EXPOSE 5000

# 7. Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
