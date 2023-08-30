# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the application files into the container
COPY . .

# Expose the port your Flask app will run on
EXPOSE 80

# Set the command to run your application
CMD ["python", "app.py"]
