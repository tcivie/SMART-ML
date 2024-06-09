FROM --platform=linux/amd64 ubuntu:latest
LABEL authors="gleb&omer"

ENV PORT 8080

# Install SUMO and system dependencies
RUN apt-get update && \
    apt-get install -y \
    sumo \
    sumo-tools \
    sumo-doc \
    python3 \
    python3-pip \
    python3-venv \
    build-essential

# Create a directory for the app
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Create and activate a virtual environment
RUN python3 -m venv /app/venv

# Upgrade pip inside the virtual environment and install dependencies
RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Set environment variables for SUMO and Python
ENV SUMO_HOME /usr/share/sumo
ENV PATH="/app/venv/bin:$PATH"

# Expose the port
EXPOSE $PORT

# Command to run the application
CMD ["python3", "init.py"]