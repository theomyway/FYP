# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the main project folder contents into the container at /app
COPY ./App /app

# Set the environment variable for Flask
ENV FLASK_APP=main.py

# Set the environment variable for production environment
ENV FLASK_ENV=production

# Make port 80 available to the world outside this container
EXPOSE 5000

# Define the command to start the app
CMD ["python", "main.py"]
