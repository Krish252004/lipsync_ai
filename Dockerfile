# Use a suitable base image with CUDA for GPU acceleration
# Choose a tag that matches your PyTorch version's CUDA compatibility
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# ffmpeg is crucial for opencv-python to process videos
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker's caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copy the rest of your application code
COPY . .

# Ensure src directory is in PYTHONPATH for internal imports
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Define a default command to run, for example, a training script
# This can be easily overridden when you run the container
CMD ["python", "src/training/train.py"] 