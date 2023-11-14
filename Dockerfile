# Use Ubuntu as the base image
FROM ubuntu:latest

# Update package lists
RUN apt-get update && \
    apt-get install -y software-properties-common

# Add deadsnakes PPA for latest Python versions
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10.9
RUN apt-get update && \
    apt-get install -y g++ python3.10 python3.10-distutils python3.10-venv git htop

# Remove existing python3 symlink and create a new one for python3.10
RUN rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Install pip for Python 3.10
RUN apt-get install -y python3-pip

# Upgrade pip to the latest version
RUN python3 -m pip install --upgrade pip

# Create a non-root user
RUN useradd -ms /bin/bash llamauser

# Set the working directory
WORKDIR /home/llmcorp

# Copy the current directory contents into the container at /home/llmcorp
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME MyApp

# Run the application
CMD ["python", "./main.py"]
