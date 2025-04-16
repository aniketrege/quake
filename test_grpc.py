#!/usr/bin/env python3

import sys
import os
import grpc
import time

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quake.proto import quake_pb2, quake_pb2_grpc

def test_connection(address):
    try:
        print(f"Testing connection to {address}...")
        channel = grpc.insecure_channel(address)
        stub = quake_pb2_grpc.QuakeServiceStub(channel)
        
        # Try to get stats
        request = quake_pb2.GetStatsRequest()
        response = stub.GetStats(request)
        
        print(f"Successfully connected to {address}")
        return True
    except Exception as e:
        print(f"Failed to connect to {address}: {str(e)}")
        return False

def main():
    # Test with container names
    addresses = [
        "quake-server-1:50051",
        "quake-server-2:50051",
        "quake-server-3:50051"
    ]
    
    print("Testing gRPC connections...")
    for addr in addresses:
        test_connection(addr)
        time.sleep(1)  # Small delay between tests

if __name__ == "__main__":
    main() 