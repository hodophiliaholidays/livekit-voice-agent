FROM python:3.11-slim

# Install system dependencies for av (PyAV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    && apt-get clean


# Set workdir
WORKDIR /app

# Copy files and install Python dependencies
COPY . .
# Install system dependencies including git
RUN apt-get update && apt-get install -y git \
    && pip install --upgrade pip \
    && pip install --use-deprecated=legacy-resolver -r requirements.txt

COPY file.env /app/file.env


CMD ["python", "main.py"]
