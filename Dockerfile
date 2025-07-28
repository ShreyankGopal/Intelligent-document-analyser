# Use official Python 3.11 base image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload the sentence-transformers model to cache it inside the image
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').save('./cached_model')"

# Use local path in your script
ENV TRANSFORMERS_OFFLINE=1

# Copy necessary project files
COPY Extract_Section.py .
COPY heading_classifier_with_font_count_norm_textNorm_5.pkl .

# Set the default command to run your script
CMD ["python3", "Extract_Section.py"]