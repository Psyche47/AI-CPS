FROM busybox:latest

LABEL maintainer="Emmanuel Gomes"

# Creating the directory
RUN mkdir -p /tmp/activationBase/

# Copy the CSV file into the container
COPY activation_data.csv /tmp/activationBase/

# Copy the Readme.md with required documentation
COPY README.md /tmp/activationBase/
