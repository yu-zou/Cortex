FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=http://192.168.104.167:7897
ENV https_proxy=http://192.168.104.167:7897
RUN apt-get update && apt-get install -y \
    build-essential cmake git libopenblas-dev liblapack-dev \
    python3 python3-pip python3-numpy \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install faiss-cpu
WORKDIR /cortex
