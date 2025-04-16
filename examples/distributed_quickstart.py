#!/usr/bin/env python
"""
Distributed Quake Example
=======================

This example demonstrates how to use the distributed version of Quake:
- Connecting to multiple Docker containers running Quake servers
- Building an index on each server
- Executing distributed search queries using both broadcast and partition strategies
- Adding and removing vectors from all servers

Usage:
    # First start the Docker containers
    docker-compose up -d
    
    # Then run this script (inside the client container)
    docker-compose run quake-client
"""

import time
import torch
import numpy as np

from quake import IndexBuildParams, QuakeIndex, SearchParams
from quake.datasets.ann_datasets import load_dataset
from quake.utils import compute_recall
from quake.index_wrappers.quake_client import QuakeClient

def main():
    print("=== Distributed Quake Example ===")
    
    # Load a sample dataset (sift1m dataset as an example)
    dataset_name = "sift1m"
    print("Loading %s dataset..." % dataset_name)
    vectors, queries, gt = load_dataset(dataset_name)
    
    # Use a subset of the queries for this example
    ids = torch.arange(vectors.size(0))
    nq = 100
    queries = queries[:nq]
    gt = gt[:nq]
    
    # Create client connecting to Docker containers
    server_addresses = [
        "quake-server-1:50051",  # Using Docker container name
        "quake-server-2:50051",  # Using Docker container name
        "quake-server-3:50051",  # Using Docker container name
    ]
    
    # Test both search strategies
    for strategy in ["broadcast", "partition"]:
        print(f"\n=== Testing {strategy} search strategy ===")
        client = QuakeClient(server_addresses, search_strategy=strategy)
        
        ######### Build the index on all servers #########
        build_params = IndexBuildParams()
        build_params.nlist = 1024
        build_params.metric = "l2"
        print(
            "Building index with num_clusters=%d over %d vectors of dimension %d..."
            % (build_params.nlist, vectors.size(0), vectors.size(1))
        )
        start_time = time.time()
        client.build(vectors, build_params.nlist, build_params.metric, ids)
        end_time = time.time()
        print(f"Build time: {end_time - start_time:.4f} seconds")
        
        ######### Search the index #########
        # Set up search parameters
        search_params = SearchParams()
        search_params.k = 10
        search_params.nprobe = 10
        
        print(
            "Performing distributed search of %d queries with k=%d and nprobe=%d..."
            % (queries.size(0), search_params.k, search_params.nprobe)
        )
        start_time = time.time()
        search_result = client.search(queries, search_params.k, search_params.nprobe)
        end_time = time.time()
        recall = compute_recall(search_result[0], gt, search_params.k)
        
        print(f"Mean recall: {recall.mean().item():.4f}")
        print(f"Search time: {end_time - start_time:.4f} seconds")
        
        ######### Remove vectors from all servers #########
        n_remove = 100
        print("Removing %d vectors from all servers..." % n_remove)
        remove_ids = torch.arange(0, n_remove)
        start_time = time.time()
        client.remove(remove_ids)
        end_time = time.time()
        print(f"Remove time: {end_time - start_time:.4f} seconds")
        
        ######### Add vectors to all servers #########
        n_add = 100
        print("Adding %d vectors to all servers..." % n_add)
        add_ids = torch.arange(vectors.size(0), vectors.size(0) + n_add)
        add_vectors = torch.randn(n_add, vectors.size(1))
        
        start_time = time.time()
        client.add(add_vectors, add_ids)
        end_time = time.time()
        print(f"Add time: {end_time - start_time:.4f} seconds")
        
        ######### Perform maintenance on all servers #########
        print("Performing maintenance on all servers...")
        start_time = time.time()
        client.maintenance()
        end_time = time.time()
        print(f"Maintenance time: {end_time - start_time:.4f} seconds")
        
if __name__ == "__main__":
    main() 