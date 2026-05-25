FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    Pillow \
    numpy \
    redis \
    opencv-python \
    huggingface-hub

RUN pip3 install git+https://github.com/huggingface/transformers    

# Copy code
COPY main.py /app/

# Create cache directory
RUN mkdir -p /app/data/cache/sam3

EXPOSE 2128

ENV PYTHONUNBUFFERED=1

CMD ["python3", "main.py"]