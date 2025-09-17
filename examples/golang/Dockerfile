# Simple Dockerfile to build and run the Go artifact example without a local toolchain
FROM golang:1.22-bullseye as builder

ARG ORT_VERSION=1.18.0
WORKDIR /opt

# Fetch ONNX Runtime prebuilt CPU package
RUN curl -fsSL -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && tar -xzf ort.tgz \
    && mv onnxruntime-linux-x64-${ORT_VERSION} onnxruntime \
    && rm ort.tgz

ENV CGO_CFLAGS="-I/opt/onnxruntime/include"
ENV CGO_LDFLAGS="-L/opt/onnxruntime/lib -lonnxruntime"
ENV LD_LIBRARY_PATH="/opt/onnxruntime/lib"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy only the Go example source
COPY examples/golang/main.go .

# Initialize module and fetch microsoft binding
RUN go mod init gaeilge_morph_go_example \
    && go env -w GOPROXY=https://proxy.golang.org,direct \
    && go get github.com/microsoft/onnxruntime-go@latest \
    && go mod tidy \
    && go build -o /usr/local/bin/gaeilge_morph_go .

FROM debian:bullseye-slim
WORKDIR /app

# Copy ONNX Runtime runtime libs and the built binary
COPY --from=builder /opt/onnxruntime /opt/onnxruntime
COPY --from=builder /usr/local/bin/gaeilge_morph_go /usr/local/bin/gaeilge_morph_go

# Copy only the minimal resources and artifacts required at runtime from build context
COPY artifacts/onnx artifacts/onnx
COPY rust/morphology_runtime/resources rust/morphology_runtime/resources

ENV LD_LIBRARY_PATH="/opt/onnxruntime/lib"

ENTRYPOINT ["/usr/local/bin/gaeilge_morph_go"]
# Example usage:
# docker build -t go-analyze -f examples/golang/Dockerfile .
# docker run --rm go-analyze --prefer-lexicon Is maidin bhreá í

