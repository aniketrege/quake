import grpc
from concurrent import futures
import torch
from typing import Optional, Tuple, Union

from quake import QuakeIndex, IndexBuildParams, SearchParams
from quake.index_wrappers.wrapper import IndexWrapper
from quake.proto import quake_pb2_grpc

class QuakeServer(IndexWrapper):
    """A server implementation of the Quake index that handles remote requests."""
    
    def __init__(self, port: int = 50051, num_workers: int = 1):
        self.index = None
        self.port = port
        self.num_workers = num_workers
        self.server = None
        
    def start(self):
        """Start the gRPC server."""
        from quake.index_wrappers.quake_service import QuakeService
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.num_workers))
        quake_pb2_grpc.add_QuakeServiceServicer_to_server(QuakeService(self), self.server)
        self.server.add_insecure_port(f'[::]:{self.port}')
        self.server.start()
        print(f"Server started on port {self.port}")
        
    def stop(self):
        """Stop the gRPC server."""
        if self.server:
            self.server.stop(0)
            print("Server stopped")
            
    def build(self, vectors: torch.Tensor, nc: int, metric: str = "l2", 
             ids: Optional[torch.Tensor] = None, num_workers: int = 0, 
             m: int = -1, code_size: int = 8):
        """Build the index with the given vectors."""
        assert vectors.ndim == 2
        assert nc > 0

        vec_dim = vectors.shape[1]
        metric = metric.lower()
        print(
            f"Building index with {vectors.shape[0]} vectors of dimension {vec_dim} "
            f"and {nc} centroids, with metric {metric}."
        )
        build_params = IndexBuildParams()
        build_params.metric = metric
        build_params.nlist = nc
        build_params.num_workers = num_workers

        self.index = QuakeIndex()

        if ids is None:
            ids = torch.arange(vectors.shape[0], dtype=torch.int64)

        return self.index.build(vectors, ids.to(torch.int64), build_params)
        
    def search(self, query: torch.Tensor, k: int, nprobe: int = 1, 
              batched_scan: bool = False, recall_target: float = -1,
              k_factor: float = 4.0, use_precomputed: bool = True,
              initial_search_fraction: float = 0.05,
              recompute_threshold: float = 0.1,
              aps_flush_period_us: int = 50,
              n_threads: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search the index for the k-nearest neighbors of the query vectors."""
        search_params = SearchParams()
        search_params.nprobe = nprobe
        search_params.recall_target = recall_target
        search_params.use_precomputed = use_precomputed
        search_params.batched_scan = batched_scan
        search_params.initial_search_fraction = initial_search_fraction
        search_params.recompute_threshold = recompute_threshold
        search_params.aps_flush_period_us = aps_flush_period_us
        search_params.k = k
        search_params.num_threads = n_threads

        result = self.index.search(query, search_params)
        return result.ids, result.distances
        
    def add(self, vectors: torch.Tensor, ids: Optional[torch.Tensor] = None, 
           num_threads: int = 0):
        """Add vectors to the index."""
        assert self.index is not None
        assert vectors.ndim == 2

        if ids is None:
            curr_id = self.n_total()
            ids = torch.arange(curr_id, curr_id + vectors.shape[0], dtype=torch.int64)

        return self.index.add(vectors, ids)
        
    def remove(self, ids: torch.Tensor):
        """Remove vectors from the index."""
        assert self.index is not None
        assert ids.ndim == 1
        return self.index.remove(ids)
        
    def maintenance(self):
        """Perform maintenance on the index."""
        return self.index.maintenance()
        
    def save(self, filename: str):
        """Save the index to a file."""
        self.index.save(str(filename))
        
    def load(self, filename: str, n_workers: int = 0, use_numa: bool = False,
            verbose: bool = False, verify_numa: bool = False,
            same_core: bool = True, use_centroid_workers: bool = False,
            use_adaptive_n_probe: bool = False):
        """Load the index from a file."""
        print(
            f"Loading index from {filename}, with {n_workers} workers, use_numa={use_numa}, verbose={verbose}, "
            f"verify_numa={verify_numa}, same_core={same_core}, use_centroid_workers={use_centroid_workers}"
        )
        self.index = QuakeIndex()
        self.index.load(str(filename), n_workers)
        
    def n_total(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal()
        
    def d(self) -> int:
        """Return the dimension of the vectors in the index."""
        return self.index.d()
        
    def index_state(self) -> dict:
        """Return the state of the index."""
        return {
            "n_list": self.index.nlist(),
            "n_total": self.index.ntotal(),
        } 

    def centroids(self) -> Union[torch.Tensor, None]:
        """Get centroids from first server"""
        # Note: This is a placeholder implementation since the gRPC service
        # doesn't currently support getting centroids. In a real implementation,
        # we would add a GetCentroids RPC method to the service.
        return None 