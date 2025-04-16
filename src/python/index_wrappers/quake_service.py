import time
import torch
import numpy as np
from concurrent import futures
import grpc
from typing import TYPE_CHECKING
from quake.datasets.ann_datasets import load_dataset

from quake.proto import quake_pb2, quake_pb2_grpc

if TYPE_CHECKING:
    from quake.index_wrappers.quake_server import QuakeServer

class QuakeService(quake_pb2_grpc.QuakeServiceServicer):
    """gRPC service implementation for the Quake index server."""
    
    def __init__(self, server: 'QuakeServer'):
        self.server = server
        
    def BuildIndex(self, request, context):
        """Build the index with vectors from the specified dataset."""
        try:
            # Load the dataset
            print(f"Loading dataset {request.dataset_name} from {request.dataset_path}...")
            vectors, queries, gt = load_dataset(
                name=request.dataset_name,
                download_dir=request.dataset_path,
                overwrite_download=False
            )
            
            # Build the index
            start_time = time.time()
            self.server.build(vectors, request.nlist, request.metric)
            build_time = time.time() - start_time
            
            return quake_pb2.BuildIndexResponse(
                success=True,
                build_time=build_time,
                num_vectors=vectors.shape[0],
                dimension=vectors.shape[1]
            )
        except Exception as e:
            return quake_pb2.BuildIndexResponse(
                success=False,
                error_message=str(e)
            )
            
    def Search(self, request, context):
        """Search for nearest neighbors."""
        try:
            # Convert flattened queries to torch tensor
            queries = torch.tensor(request.queries, dtype=torch.float32)
            queries = queries.reshape(request.num_queries, request.dimension)
            
            # Perform search
            start_time = time.time()
            ids, distances = self.server.search(
                queries,
                k=request.k,
                nprobe=request.nprobe,
                recall_target=request.recall_target,
                batched_scan=request.batched_scan
            )
            search_time = time.time() - start_time
            
            # Flatten results
            flat_ids = ids.numpy().flatten().tolist()
            flat_distances = distances.numpy().flatten().tolist()
            
            return quake_pb2.SearchResponse(
                success=True,
                ids=flat_ids,
                distances=flat_distances,
                search_time=search_time
            )
        except Exception as e:
            return quake_pb2.SearchResponse(
                success=False,
                error_message=str(e)
            )
            
    def AddVectors(self, request, context):
        """Add vectors to the index."""
        try:
            # Convert flattened vectors to torch tensor
            vectors = torch.tensor(request.vectors, dtype=torch.float32)
            vectors = vectors.reshape(request.num_vectors, request.dimension)
            
            # Convert IDs if provided
            ids = None
            if request.ids:
                ids = torch.tensor(request.ids, dtype=torch.int64)
            
            # Add vectors
            start_time = time.time()
            self.server.add(vectors, ids)
            add_time = time.time() - start_time
            
            return quake_pb2.AddVectorsResponse(
                success=True,
                add_time=add_time
            )
        except Exception as e:
            return quake_pb2.AddVectorsResponse(
                success=False,
                error_message=str(e)
            )
            
    def RemoveVectors(self, request, context):
        """Remove vectors from the index."""
        try:
            # Convert IDs to torch tensor
            ids = torch.tensor(request.ids, dtype=torch.int64)
            
            # Remove vectors
            start_time = time.time()
            self.server.remove(ids)
            remove_time = time.time() - start_time
            
            return quake_pb2.RemoveVectorsResponse(
                success=True,
                remove_time=remove_time
            )
        except Exception as e:
            return quake_pb2.RemoveVectorsResponse(
                success=False,
                error_message=str(e)
            )
            
    def Maintenance(self, request, context):
        """Perform maintenance on the index."""
        try:
            start_time = time.time()
            self.server.maintenance()
            maintenance_time = time.time() - start_time
            
            return quake_pb2.MaintenanceResponse(
                success=True,
                maintenance_time=maintenance_time
            )
        except Exception as e:
            return quake_pb2.MaintenanceResponse(
                success=False,
                error_message=str(e)
            )
            
    def GetStats(self, request, context):
        """Get server statistics."""
        try:
            state = self.server.index_state()
            
            return quake_pb2.GetStatsResponse(
                success=True,
                num_vectors=state["n_total"],
                dimension=self.server.d(),
                num_clusters=state["n_list"],
                memory_usage=0.0  # TODO: Implement memory usage tracking
            )
        except Exception as e:
            return quake_pb2.GetStatsResponse(
                success=False,
                error_message=str(e)
            ) 