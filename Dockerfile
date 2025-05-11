FROM python:3.12.5-slim

WORKDIR /app

# Copy all local files into the container
COPY . /app/

# Update packages and install build tools
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    llvm \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Docker/Portainer healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/healthz || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=512", "--client.toolbarMode=minimal"]
