FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_ENABLE_HF_TRANSFER=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl unzip libgomp1 libxml2-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/pace
COPY pace.tar /opt/pace.tar
RUN tar -xf /opt/pace.tar -C /opt/pace && rm /opt/pace.tar \
    && python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && bash evaluations/install.sh \
    && test -x /root/.planutils/packages/val/bin/validate
ENTRYPOINT ["python", "evaluations/score_new_model.py"]
