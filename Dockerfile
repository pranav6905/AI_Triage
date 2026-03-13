# Use official Python lightweight image
FROM python:3.10-slim

# Set the working directory
WORKDIR /code

# Install system dependencies (PyMuPDF and Vision sometimes need these)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy the requirements file and install dependencies
COPY --chown=user:user requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of your application code
COPY --chown=user:user . /code/

# Expose the specific port Hugging Face uses
EXPOSE 7860

# Command to run the FastAPI app on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]