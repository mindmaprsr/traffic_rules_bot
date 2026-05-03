# Use Python base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "traffic_bot.py", "--server.port=8501", "--server.address=0.0.0.0"]