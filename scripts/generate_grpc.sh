#!/bin/bash

# Create proto directory if it doesn't exist
mkdir -p /quake/src/python/proto

# Generate Python gRPC files
cd /quake && \
python3 -m grpc_tools.protoc \
    -I/quake \
    --python_out=/quake/src/python/proto \
    --grpc_python_out=/quake/src/python/proto \
    /quake/src/python/proto/quake.proto

# Fix imports in generated files if they exist
if [ -f "/quake/src/python/proto/quake_pb2_grpc.py" ]; then
    sed -i 's/import quake_pb2/from . import quake_pb2/g' /quake/src/python/proto/quake_pb2_grpc.py
fi 