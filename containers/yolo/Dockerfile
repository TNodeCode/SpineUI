# This is the base image we build our image on
FROM bitnami/pytorch:2.2.1

# Get root priviledges
USER root

# Install necessary CV libraries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Install FastAPI and Ultralytics libraries
RUN pip install ultralytics "fastapi[all]" dill

# Copy Python files
COPY ./ ./

# Command to run the app
CMD uvicorn api:app --host 0.0.0.0 --port 80
