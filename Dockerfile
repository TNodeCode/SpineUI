# Base image
FROM bitnami/python:3.12

# Install CV libraries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the requirements file
COPY ./requirements.txt ./

# Install necessary libraries
RUN pip install -r requirements.txt

# Copy source code
COPY ./ ./

# Command to run the app
CMD ["streamlit", "run", "App.py"]