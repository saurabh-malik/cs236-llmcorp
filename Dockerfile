# Use the Deep Learning Containers image with CUDA 11.3 and Python 3.10 as the base
FROM gcr.io/deeplearning-platform-release/base-cu113

# Create a non-root user (optional but recommended)
RUN useradd -ms /bin/bash llamauser

# Set the working directory
WORKDIR /home/llmcorp

# Copy the current directory contents into the container at /home/llmcorp
COPY . .

# Since Python, pip, and many common packages are pre-installed in the base image,
# we only need to install additional dependencies specified in requirements.txt.
# Note that you should remove any CUDA-related packages from requirements.txt,
# as they are already included in the base image.
RUN python -m pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME MyApp

# Run the application
CMD ["python", "./main.py"]