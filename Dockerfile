# Lightweight Python base image
FROM python:3.11-slim

# avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies required for OpenCV + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install Python packages
# - install CPU-only PyTorch wheel first (saves space if you don't need CUDA)
# - then install the rest of requirements
RUN pip install --no-cache-dir torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Optional healthcheck (uses curl)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
