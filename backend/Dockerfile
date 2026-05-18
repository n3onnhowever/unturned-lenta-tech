FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
COPY lenta-hackathon-main/requirements_backend.txt ./lenta-hackathon-main/requirements_backend.txt
RUN pip install --no-cache-dir -r requirements.txt -r lenta-hackathon-main/requirements_backend.txt \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless \
    && pip install --no-cache-dir --force-reinstall --no-deps "opencv-contrib-python-headless>=4.10"

COPY app ./app
COPY video_preprocess.py ./video_preprocess.py
COPY lenta-hackathon-main ./lenta-hackathon-main

EXPOSE 8000
