# Use an official Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install them early (for caching)
COPY requirements.txt .

# Install required system packages (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean

# Copy the entire project into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run your app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

