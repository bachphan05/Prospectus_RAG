# 1. Start with a lightweight Python 3.12 image (matches your README requirement)
FROM python:3.12-slim
# 2. Set environment variables to prevent Python from writing .pyc files
# and to ensure logs are output to the terminal immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3.Setup working directory inside the container
WORKDIR /app

# 4. Install system dependencies required for psycopg2 and other libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libxcb1 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy your requirements.txt first (for better caching)
COPY requirements.txt /app/

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your backend code
COPY . /app/

# 8. The command to run your server (we usually override this in docker-compose, but this is a good default)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]