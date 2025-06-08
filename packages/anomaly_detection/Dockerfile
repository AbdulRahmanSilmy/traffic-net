# Use a lightweight base image compatible with Raspberry Pi (ARM-based architecture)
FROM python:3.10-slim

# Install Poetry
RUN pip install poetry


# Set working directory
WORKDIR /app

# Copy your requirements file to install dependencies
COPY anomaly_detection/ anomaly_detection/
COPY common_utils/ common_utils/

# Change directory to data_ingestion
WORKDIR /app/anomaly_detection

# Install necessary Python libraries for the script
RUN poetry install --no-root

# Run the script indefinitely
CMD ["poetry", "run", "python", "-m", "anomaly_detection"]
