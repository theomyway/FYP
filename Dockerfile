# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set the environment variable for Flask
ENV FLASK_APP=main.py


# Expose the port that the app will run on
EXPOSE 5000

CMD ["python", "main.py"]
