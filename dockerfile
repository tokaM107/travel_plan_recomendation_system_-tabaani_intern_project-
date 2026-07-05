# Use official Python image
FROM python:3.11-slim

# Set work directory inside container
WORKDIR /app

# Copy requirements first (caching advantage)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the FastAPI app by default
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
