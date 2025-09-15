# Use official Python image
FROM python:3.11

# Set work directory inside container
WORKDIR /app

# Copy requirements first (caching advantage)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run your script by default
CMD ["python", "main.py"]
