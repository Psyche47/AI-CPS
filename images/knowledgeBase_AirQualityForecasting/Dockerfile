FROM busybox:latest

LABEL maintainer="Emmanuel Gomes"

# Create the directory within the container to hold the AI model
RUN mkdir -p /tmp/knowledgeBase/

# Copying the AI/OLS model into the image
COPY currentAiSolution.keras /tmp/knowledgeBase/
COPY currentOlsSolution.pkl /tmp/knowledgeBase/

COPY README.md /tmp/knowledgeBase/

CMD ["sh"]
