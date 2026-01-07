FROM vastai/pytorch:2.8.0-cuda-12.6.3-py311-24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-21-jdk \
        python3-dev \
        build-essential \
        git \
        curl \
        wget \
        unzip \
        ca-certificates \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYSERINI_CACHE=/workspace/.cache/pyserini
RUN mkdir -p "${PYSERINI_CACHE}"

COPY start.sh /usr/local/bin/start.sh
RUN sed -i 's/\r$//' /usr/local/bin/start.sh \
    && chmod +x /usr/local/bin/start.sh

EXPOSE 8888

CMD ["bash", "/usr/local/bin/start.sh"]
