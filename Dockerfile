# Use Python 3.12 slim base image
FROM python:3.12.5-slim

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app/

# Install system dependencies required for Python packages like Pillow and SHAP
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    tk-dev \
    tcl-dev \
    python3-dev \
    curl \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit’s default port
EXPOSE 8501

# Healthcheck (optional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the app with specific streamlit options
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=512", "--client.toolbarMode=minimal"]
