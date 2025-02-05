FROM python:3.10-slim

# Create the /tmp/codeBase/ directory
RUN mkdir -p /tmp/codeBase/

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    statsmodels \
    tensorflow

# Copy Python scripts
COPY ANNActivation.py /tmp/codeBase/
COPY OLSActivition.py /tmp/codeBase/

