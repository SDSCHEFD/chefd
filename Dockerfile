# Use an official Python runtime as a parent image
FROM python:3.5-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

RUN apt-get install bash -y

USER root

# Install any needed packages specified in requirements.txt
RUN pip install pandas numpy sklearn scipy

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

CMD ["tail", "-f", "/dev/null"]
