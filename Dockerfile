# Lightweight Python base image
FROM python:3.11-slim

# avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy only requirements first (layer caching)
COPY requirements.txt .

# Install system libs required by OpenCV and imaging libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Use Cloud Run conventional port 8080
ENV PORT=8080
EXPOSE 8080

# Run Streamlit on 0.0.0.0 so Cloud Run can route traffic
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
