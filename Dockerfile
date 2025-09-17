# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system deps and Rust
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev git unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (stable)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> /etc/profile
ENV PATH=/root/.cargo/bin:${PATH}

# Set workdir
WORKDIR /workspace

# Copy project files
COPY pyproject.toml README.md Makefile ./
COPY src ./src
COPY scripts ./scripts
COPY rust ./rust
COPY RESEARCH.md LICENSE ./

# Install Python deps
RUN pip install --upgrade pip && pip install -e .[dev]

# Default command
CMD ["bash"]
