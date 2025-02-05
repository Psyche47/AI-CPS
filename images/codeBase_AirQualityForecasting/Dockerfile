FROM ubuntu:latest

# Create the /tmp/codeBase/ directory
RUN mkdir -p /tmp/codeBase/

# Install system packages and Python 3
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip3 install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    statsmodels \
    tensorflow

# Copy Python scripts
COPY ANNActivation.py /tmp/codeBase/
COPY OLSActivition.py /tmp/codeBase/

