FROM ubuntu:latest
LABEL authors="gleb&omer"

ENV PORT 8080

# Install SUMO. This will include SUMO itself, tools and documentation.
RUN apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc \
    python3 python3-pip

COPY requirements.txt /app/requirements.txt

# Install TraCI for Python
RUN pip3 install -r /app/requirements.txt

# Set the SUMO_HOME environment variable
# Replace '/usr/share/sumo' with the actual SUMO installation path if different
ENV SUMO_HOME /usr/share/sumo
ENV SUMO_CONF_FILE run.sumocfg

# Set the working directory inside the container to /sumo-data
WORKDIR /app

# Copy your Python script into the Docker image
COPY init.py /app/init.py

# Copy all sumo files into the Docker image
COPY scenarios /app/scenarios

COPY sumo_sim /app/sumo_sim
COPY utils /app/utils

# Expose a port if you need to access the SUMO environment from outside the container
EXPOSE $PORT

# Run your TraCI script (replace 'your_traci_script.py' with your actual script name)
CMD ["python3", "init.py"]
