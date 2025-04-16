FROM ubuntu:24.04

WORKDIR /

# Disable caching and broken proxy
RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

RUN apt clean && rm -rf /var/lib/apt/lists/*


# Install required packages
RUN apt update && apt install -y git python3-pip cmake libblas-dev liblapack-dev libnuma-dev libgtest-dev

RUN pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . /quake

# Fix arm libgfortran name issue
RUN arch=$(uname -m) && \
if [ "$arch" = "aarch64" ]; then \
    echo "Running on ARM64"; \
    ln -s /usr/lib/aarch64-linux-gnu/libgfortran.so.5.0.0 /usr/lib/aarch64-linux-gnu/libgfortran-4435c6db.so.5.0.0 ; \
elif [ "$arch" = "x86_64" ]; then \
    echo "Running on AMD64"; \
else \
    echo "Unknown architecture: $arch"; \
fi

WORKDIR /quake

# Install gRPC dependencies
RUN pip install --break-system-packages grpcio grpcio-tools protobuf

# gRPC files already generated
# RUN chmod +x scripts/generate_grpc.sh && ./scripts/generate_grpc.sh

# Build Quake
RUN mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
        -DQUAKE_ENABLE_GPU=OFF \
        -DQUAKE_USE_NUMA=OFF \
        -DQUAKE_USE_AVX512=OFF \
        -DBUILD_TESTS=ON \
        -DQUAKE_SET_ABI_MODE=OFF .. \
    && make bindings -j$(nproc) \
    && make quake_tests -j$(nproc)

# Install Python package
RUN pip install --no-use-pep517 --break-system-packages .

# Create data directory
RUN mkdir -p /quake/data

# Expose the gRPC port
EXPOSE 50051

# Command to start the server
CMD ["python3", "-m", "quake.index_wrappers.quake_server", "--port", "50051"] 

