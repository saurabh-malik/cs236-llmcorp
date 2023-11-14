# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/llmcorp

# Copy the current directory contents into the container at /usr/src/app
COPY . .


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Make port 5000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME MyApp

# Run the application
CMD ["python", "./main.py"]
