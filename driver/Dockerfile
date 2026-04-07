FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=http://192.168.104.167:7897
ENV https_proxy=http://192.168.104.167:7897
RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /cortex
