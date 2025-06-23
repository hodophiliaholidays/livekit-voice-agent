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
    libavresample-dev \
    && apt-get clean

# Set workdir
WORKDIR /app

# Copy files and install Python dependencies
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "main.py"]
