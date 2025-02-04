FROM ubuntu:latest

RUN mkdir -p /tmp/codeBase/

# Install system packages and Python 3
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the Python packages
RUN pip3 install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    statsmodels \
    tensorflow

COPY ANNModelCreation.py /tmp/codeBase/
COPY ANNModelTest.py /tmp/codeBase/
COPY OLSModelCreation.py /tmp/codeBase/
COPY OLSModelTest.py /tmp/codeBase/


CMD ["python3"]
