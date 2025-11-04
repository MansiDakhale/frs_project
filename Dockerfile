# Dockerfile

# --- Stage 1: Build Stage ---
# We use a full-fat Python image to build our dependencies,
# as some (like numpy, opencv) need to be compiled.
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /app

# Install system-level dependencies required for OpenCV and others
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
# We install to a special "target" directory
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --target=/app/wheels -r requirements.txt


# --- Stage 2: Final Production Stage ---
# We use a slim base image for the final container.
FROM python:3.9-slim

# Set a non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /home/appuser/app

# Copy the installed python packages from the builder stage
COPY --from=builder /app/wheels /home/appuser/app/

# Copy the application code
COPY . .

# Create the gallery directory so the app can save new images
RUN mkdir -p gallery

# Expose the port the app runs on
EXPOSE 8000

# Set the PYTHONPATH to include our installed wheels
ENV PYTHONPATH=/home/appuser/app
# Define the command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]