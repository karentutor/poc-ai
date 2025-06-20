FROM python:3.12-slim

# 1. system deps
RUN apt-get update && apt-get -y install libgl1 ghostscript && rm -rf /var/lib/apt/lists/*

# 2. copy code + index
WORKDIR /app
COPY app/ app/
COPY static/ static/

COPY widget.faiss widget_names.pkl .

# 3. install Python deps
RUN pip install --no-cache-dir \
        fastapi uvicorn[standard] python-multipart \
        pymupdf faiss-cpu sentence_transformers

# 4. run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
