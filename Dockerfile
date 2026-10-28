# Start from a small official Python image
FROM python:3.12-slim

# All commands below run inside /app
WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model
COPY app ./app
COPY model ./model

# The API listens on this port
EXPOSE 8000

# Start the FastAPI server. 0.0.0.0 makes it reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]