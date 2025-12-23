FROM python:3.13.2

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libopus-dev \
    libffi-dev \
    pkg-config \
    curl \
    procps \
    # Các gói cần để build SWIG 3.0.12
    libpcre3-dev \
    automake \
    autoconf \
    bison \
    wget \
    tar \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/swig/swig/archive/refs/tags/v3.0.12.tar.gz && \
    tar -xzvf v3.0.12.tar.gz && \
    cd swig-3.0.12 && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm v3.0.12.tar.gz

RUN python -m venv $VIRTUAL_ENV

COPY pyproject.toml ./
COPY uv.lock* ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv && \
    uv pip install -r pyproject.toml

RUN pip install --no-cache-dir python-ESL

RUN mkdir -p /app/logs

COPY . .

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD pgrep -f "server-worker.py" || exit 1

# Default command
CMD ["python", "server-worker.py", "start"]