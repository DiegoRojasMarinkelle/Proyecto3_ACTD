# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# Install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the app files
COPY Tablero.py /

# Final configuration
EXPOSE 8050
CMD ["python", "Tablero.py"]
