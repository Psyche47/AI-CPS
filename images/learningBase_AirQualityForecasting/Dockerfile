FROM busybox

LABEL maintainer="Emmanuel Gomes"

# Creating the directories
RUN mkdir -p /tmp/learningBase/train/ && 
    mkdir -p /tmp/learningBase/validation/

# Copy the CSV files into the container
COPY training_data.csv /tmp/learningBase/train/
COPY test_data.csv /tmp/learningBase/validation/

# Copy the Readme.md into the container
COPY README.md /tmp/learningBase/

CMD ["sh"]