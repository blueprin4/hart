# Use RunPod's CUDA-enabled PyTorch image as the base
FROM runpod/pytorch:2.3.0-py3.10-cuda12.1.105-ubuntu20.04

WORKDIR /workspace

# Install system tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    nano \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install transformers scipy accelerate gradio spaces

# Clone your working HART repo into the image
RUN git clone https://github.com/YOUR-USERNAME/hart.git /workspace/hart

WORKDIR /workspace/hart

# Install hart as a local package
RUN pip install -e .

# Download or clone the models into the image (optional – could also mount externally)
COPY models /workspace/hart/models

# Expose Gradio UI port
EXPOSE 7860

# Start the Gradio app on launch
CMD ["python3", "app.py", "--model_path", "./models/hart-0.7b-1024px/llm", "--text_model_path", "./models/Qwen2-VL-1.5B-Instruct"]