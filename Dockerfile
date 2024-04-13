# Base image
FROM bitnami/python:3.12

# Copy the requirements file
COPY ./requirements.txt ./

# Install necessary libraries
RUN pip install -r requirements.txt

# Copy source code
COPY ./ ./

# Command to run the app
CMD ["streamlit", "run", "App.py"]