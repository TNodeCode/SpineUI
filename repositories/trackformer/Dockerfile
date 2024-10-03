# Base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Default shell to use
SHELL ["/bin/bash","-l", "-c"]

# Working directory
WORKDIR /app

# Get root priviledges
USER root

# Install necessary libraries
RUN apt-get update && apt-get install -y build-essential cmake git

# Remove default PyTorch version
RUN pip uninstall -y torch torchvision torchaudio

# Create a new conda environment
RUN conda create -y -n trackformer python==3.7.4 pip

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "trackformer", "/bin/bash", "-c"]

#RUN conda activate trackformer
RUN conda run --no-capture-output -n trackformer pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Copy requirements.txt
COPY ./requirements.txt ./requirements.txt

# Install requirements
RUN conda run --no-capture-output -n trackformer pip install -U -r requirements.txt
RUN conda run --no-capture-output -n trackformer pip install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'

# Copy all files
COPY ./ ./

# Install operators
#RUN apt-get install -y nvidia-cuda-dev nvidia-cuda-toolkit 
#RUN python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install