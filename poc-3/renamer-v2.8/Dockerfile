# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Copy environment files
COPY environment.yml /tmp/environment.yml
COPY setup_env.sh /tmp/setup_env.sh

# Create conda environment
RUN conda env create -f /tmp/environment.yml

# Make setup script executable
RUN chmod +x /tmp/setup_env.sh

# Set the working directory
WORKDIR /app

# Activate conda environment
SHELL ["conda", "run", "-n", "XENEX", "/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"] 