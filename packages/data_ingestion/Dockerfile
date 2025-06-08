# Use a lightweight base image compatible with Raspberry Pi (ARM-based architecture)
FROM python:3.9-slim

# Install Poetry
RUN pip install poetry

# Install system dependencies for OpenCV and other necessary libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your requirements file to install dependencies
COPY data_ingestion/ data_ingestion/
COPY common_utils/ common_utils/

# Change directory to data_ingestion
WORKDIR /app/data_ingestion

# Install necessary Python libraries for the script
RUN poetry install --no-root

# Run the script indefinitely
CMD ["poetry", "run", "python", "-m", "data_ingestion"]
