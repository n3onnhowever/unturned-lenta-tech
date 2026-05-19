FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY lenta-hackathon-main/requirements_backend.txt ./lenta-hackathon-main/requirements_backend.txt

RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.3.1+cpu" "torchvision==0.18.1+cpu" \
    && pip install -r requirements.txt -r lenta-hackathon-main/requirements_backend.txt \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless \
    && pip install --force-reinstall --no-deps "opencv-contrib-python-headless>=4.10"

COPY video_preprocess.py ./video_preprocess.py
COPY app ./app
COPY lenta-hackathon-main ./lenta-hackathon-main
COPY start-hf.sh ./start-hf.sh

RUN chmod +x ./start-hf.sh

EXPOSE 7860

CMD ["./start-hf.sh"]

