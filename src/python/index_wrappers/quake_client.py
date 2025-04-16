import time
import torch
import numpy as np
from typing import Optional, Tuple, Union, List
import grpc
from queue import PriorityQueue
import heapq

from quake import QuakeIndex, IndexBuildParams, SearchParams
from quake.index_wrappers.wrapper import IndexWrapper
from quake.proto import quake_pb2, quake_pb2_grpc

class QuakeClient(IndexWrapper):
    """Client for distributed Quake index system.
    
    This client can operate in two modes:
    1. Broadcast mode: Sends all queries to all servers and merges results
    2. Partition mode: Splits queries across servers and merges results
    
    Args:
        server_addresses: List of server addresses in format "host:port"
        search_strategy: Either "broadcast" or "partition"
    """
    
    def __init__(self, server_addresses: List[str], search_strategy: str = "broadcast"):
        super().__init__()
        self.server_addresses = server_addresses
        self.search_strategy = search_strategy
        if search_strategy not in ["broadcast", "partition"]:
            raise ValueError("search_strategy must be either 'broadcast' or 'partition'")
            
        # Create gRPC channels and stubs
        self.channels = []
        self.servers = []
        for address in server_addresses:
            channel = grpc.insecure_channel(address)
            self.channels.append(channel)
            self.servers.append(quake_pb2_grpc.QuakeServiceStub(channel))
            
    def build(self, vectors: torch.Tensor, nc: int, metric: str = "l2", 
             ids: Optional[torch.Tensor] = None, num_workers: int = 0, 
             m: int = -1, code_size: int = 8):
        """Build the index on all servers."""
        # Create build request
        request = quake_pb2.BuildIndexRequest(
            dataset_path="/data",  # Path where datasets are mounted in the container
            dataset_name="sift1m",  # Default dataset name
            nlist=nc,
            metric=metric
        )
        
        # Send request to all servers
        responses = []
        for server in self.servers:
            response = server.BuildIndex(request)
            responses.append(response)
            
        # Check for errors
        for response in responses:
            if not response.success:
                raise RuntimeError(f"Failed to build index: {response.error_message}")
                
    def search(
        self,
        query: torch.Tensor,
        k: int,
        nprobe: int = 1,
        recall_target: Optional[float] = None,
        max_iterations: int = 10,
        batch_size: int = 1000,
        num_threads: int = 1,
        use_gpu: bool = False,
        gpu_id: int = 0,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search for nearest neighbors across all servers.
        
        Args:
            query: Query vectors of shape (nq, d)
            k: Number of nearest neighbors to return
            nprobe: Number of clusters to probe
            recall_target: Target recall to achieve (optional)
            max_iterations: Maximum number of iterations for recall target
            batch_size: Batch size for processing queries
            num_threads: Number of threads to use
            use_gpu: Whether to use GPU
            gpu_id: GPU ID to use
            verbose: Whether to print progress
            
        Returns:
            Tuple of (ids, distances) where:
            - ids: Tensor of shape (nq, k) containing the IDs of nearest neighbors
            - distances: Tensor of shape (nq, k) containing the distances
        """
        # Convert queries to flattened format
        flat_queries = query.numpy().flatten().tolist()
        
        if self.search_strategy == "broadcast":
            # Broadcast mode: Send all queries to all servers
            request = quake_pb2.SearchRequest(
                queries=flat_queries,
                num_queries=query.shape[0],
                dimension=query.shape[1],
                k=k,
                nprobe=nprobe,
                recall_target=recall_target if recall_target is not None else 0.0,
                max_iterations=max_iterations,
                batch_size=batch_size,
                num_threads=num_threads,
                use_gpu=use_gpu,
                gpu_id=gpu_id,
                verbose=verbose,
            )
            
            # Send request to all servers in parallel
            responses = []
            for server in self.servers:
                response = server.Search(request)
                responses.append(response)
                
        else:  # partition mode
            # Partition mode: Split queries across servers
            n_servers = len(self.servers)
            queries_per_server = query.shape[0] // n_servers
            remainder = query.shape[0] % n_servers
            
            responses = []
            start_idx = 0
            for i, server in enumerate(self.servers):
                # Calculate number of queries for this server
                n_queries = queries_per_server + (1 if i < remainder else 0)
                if n_queries == 0:
                    continue
                    
                # Get queries for this server
                server_queries = flat_queries[start_idx * query.shape[1] : (start_idx + n_queries) * query.shape[1]]
                start_idx += n_queries
                
                request = quake_pb2.SearchRequest(
                    queries=server_queries,
                    num_queries=n_queries,
                    dimension=query.shape[1],
                    k=k,
                    nprobe=nprobe,
                    recall_target=recall_target if recall_target is not None else 0.0,
                    max_iterations=max_iterations,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    use_gpu=use_gpu,
                    gpu_id=gpu_id,
                    verbose=verbose,
                )
                
                response = server.Search(request)
                responses.append(response)
        
        # Merge results from all servers
        nq = query.shape[0]
        all_ids = np.zeros((nq, k), dtype=np.int64)
        all_distances = np.zeros((nq, k), dtype=np.float32)
        
        if self.search_strategy == "broadcast":
            # For broadcast mode, use priority queue to merge results
            for q in range(nq):
                pq = PriorityQueue()
                for response in responses:
                    if not response.success:
                        continue
                    for i in range(k):
                        idx = q * k + i
                        if idx < len(response.ids):
                            pq.put((response.distances[idx], response.ids[idx]))
                
                # Get top k results
                for i in range(k):
                    if not pq.empty():
                        dist, idx = pq.get()
                        all_distances[q, i] = dist
                        all_ids[q, i] = idx
                        
        else:  # partition mode
            # For partition mode, just concatenate results
            start_idx = 0
            for response in responses:
                if not response.success:
                    continue
                n_queries = response.num_queries
                end_idx = start_idx + n_queries
                all_ids[start_idx:end_idx] = np.array(response.ids).reshape(n_queries, k)
                all_distances[start_idx:end_idx] = np.array(response.distances).reshape(n_queries, k)
                start_idx = end_idx
        
        return torch.from_numpy(all_ids), torch.from_numpy(all_distances)
        
    def add(self, vectors: torch.Tensor, ids: Optional[torch.Tensor] = None, 
           num_threads: int = 0):
        """Add vectors to all servers."""
        # Convert vectors to flattened format
        flat_vectors = vectors.numpy().flatten().tolist()
        
        # Convert IDs if provided
        flat_ids = None
        if ids is not None:
            flat_ids = ids.numpy().flatten().tolist()
            
        # Create add request
        request = quake_pb2.AddVectorsRequest(
            vectors=flat_vectors,
            num_vectors=vectors.shape[0],
            dimension=vectors.shape[1],
            ids=flat_ids if flat_ids else []
        )
        
        # Send request to all servers
        responses = []
        for server in self.servers:
            response = server.AddVectors(request)
            responses.append(response)
            
        # Check for errors
        for response in responses:
            if not response.success:
                raise RuntimeError(f"Failed to add vectors: {response.error_message}")
                
    def remove(self, ids: torch.Tensor):
        """Remove vectors from all servers."""
        # Convert IDs to flattened format
        flat_ids = ids.numpy().flatten().tolist()
        
        # Create remove request
        request = quake_pb2.RemoveVectorsRequest(ids=flat_ids)
        
        # Send request to all servers
        responses = []
        for server in self.servers:
            response = server.RemoveVectors(request)
            responses.append(response)
            
        # Check for errors
        for response in responses:
            if not response.success:
                raise RuntimeError(f"Failed to remove vectors: {response.error_message}")
                
    def maintenance(self):
        """Perform maintenance on all servers."""
        request = quake_pb2.MaintenanceRequest()
        
        # Send request to all servers
        responses = []
        for server in self.servers:
            response = server.Maintenance(request)
            responses.append(response)
            
        # Check for errors
        for response in responses:
            if not response.success:
                raise RuntimeError(f"Maintenance failed: {response.error_message}")
                
    def save(self, filename: str):
        """Save is not supported in distributed mode."""
        raise NotImplementedError("Save is not supported in distributed mode")
        
    def load(self, filename: str, n_workers: int = 0, use_numa: bool = False,
            verbose: bool = False, verify_numa: bool = False,
            same_core: bool = True, use_centroid_workers: bool = False,
            use_adaptive_n_probe: bool = False):
        """Load is not supported in distributed mode."""
        raise NotImplementedError("Load is not supported in distributed mode")
        
    def n_total(self) -> int:
        """Return the total number of vectors across all servers."""
        # Get stats from first server
        request = quake_pb2.GetStatsRequest()
        response = self.servers[0].GetStats(request)
        
        if not response.success:
            raise RuntimeError(f"Failed to get stats: {response.error_message}")
            
        return response.num_vectors
        
    def d(self) -> int:
        """Return the dimension of the vectors."""
        # Get stats from first server
        request = quake_pb2.GetStatsRequest()
        response = self.servers[0].GetStats(request)
        
        if not response.success:
            raise RuntimeError(f"Failed to get stats: {response.error_message}")
            
        return response.dimension
        
    def index_state(self) -> dict:
        """Get index state from first server"""
        request = quake_pb2.GetStatsRequest()
        response = self.servers[0].GetStats(request)
        if not response.success:
            raise RuntimeError(f"Failed to get index state: {response.error}")
        return {
            "n_total": response.stats.n_total,
            "dimension": response.stats.dimension,
            "n_clusters": response.stats.n_clusters,
            "memory_usage": response.stats.memory_usage,
        }

    def centroids(self) -> Union[torch.Tensor, None]:
        """Get centroids from first server"""
        # Note: This is a placeholder implementation since the gRPC service
        # doesn't currently support getting centroids. In a real implementation,
        # we would add a GetCentroids RPC method to the service.
        return None 