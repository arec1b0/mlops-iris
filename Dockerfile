# Build stage - Install all dependencies including testing tools
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install all dependencies (including testing tools)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Runtime stage - Only runtime dependencies
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Set working directory
WORKDIR /app

# Copy requirements and install only runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code from builder stage
COPY --from=builder /app/src/ ./src/

# Create artifacts directory and set permissions
RUN mkdir -p artifacts && chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Expose port
EXPOSE 80

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]