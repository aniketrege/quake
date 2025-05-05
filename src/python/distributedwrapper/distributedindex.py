import asyncio
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Tuple

import torch
from quake import QuakeIndex, IndexBuildParams, SearchParams
from quake.distributedwrapper import distributed


class DistributedIndex:
    """
    A distributed version of QuakeIndex that supports multiple servers.
    Each server maintains a full copy of the index, and queries are distributed
    across servers for parallel processing.
    """

    def __init__(
        self,
        server_addresses: List[str],
        num_partitions: int,
        build_params_kw_args: Dict[str, Any],
        search_params_kw_args: Dict[str, Any],
        use_kmeans: bool = False,
    ):
        """
        Initialize the DistributedIndex with a list of server addresses.

        Args:
            server_addresses: List of server addresses in the format "host:port"
            num_partitions: Number of partitions to split the data into
            build_params_kw_args: Keyword arguments for building the index
            search_params_kw_args: Keyword arguments for search parameters
            use_kmeans: Whether to use k-means clustering for partitioning (default: False)
        """
        if not server_addresses:
            raise ValueError("At least one server address must be provided")

        self.server_addresses = server_addresses
        self.build_params_kw_args = build_params_kw_args
        self.search_params_kw_args = search_params_kw_args
        self.use_kmeans = use_kmeans
        self.results_list = []  # Store search results for analysis
        self.top_k_server_counts = defaultdict(int)  # Track which servers contributed to top k results
        self.stats = defaultdict(int)

        self.build_params: List[IndexBuildParams] = []
        self._initialize_build_params()

        self.indices: List[QuakeIndex] = []
        self._initialize_indices()

        self.search_params: List[SearchParams] = []
        self._initialize_search_params()

        self.k = self.search_params_kw_args["k"]

        # TODO if there are leftover servers, replicate most commonly accessed partitions
        assert (
            len(self.server_addresses) % num_partitions == 0
        ), "Number of servers must be divisible by number of partitions"

        self.num_partitions = num_partitions

    def _initialize_build_params(self):
        """Initialize IndexBuildParams instances for each server."""
        for address in self.server_addresses:
            params = distributed(IndexBuildParams, address)
            params.import_module(package="quake", item="IndexBuildParams")
            params.instantiate()
            params.nlist = self.build_params_kw_args["nlist"]
            params.metric = self.build_params_kw_args["metric"]
            self.build_params.append(params)

    def _initialize_indices(self):
        """Initialize QuakeIndex instances for each server."""
        for address in self.server_addresses:
            index = distributed(QuakeIndex, address)
            index.import_module(package="quake", item="QuakeIndex")
            index.register_function("build")
            index.register_function("search")
            index.register_function("add")
            index.register_function("remove")
            index.instantiate()
            self.indices.append(index)

    def _initialize_search_params(self):
        """Initialize SearchParams instances for each server."""
        for address in self.server_addresses:
            params = distributed(SearchParams, address)
            params.import_module(package="quake", item="SearchParams")
            params.instantiate()
            params.k = self.search_params_kw_args["k"]
            params.nprobe = self.search_params_kw_args["nprobe"]
            self.search_params.append(params)

    def _prepartition_vectors(self, vectors: torch.Tensor, ids: torch.Tensor):
        """
        Prepartition the vectors and ids into num_partitions.
        If use_kmeans is True, uses FAISS k-means clustering to partition the vectors.
        Otherwise, splits the vectors evenly across partitions.

        Args:
            vectors: Tensor of vectors to partition
            ids: Tensor of vector IDs to partition

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Lists of partitioned vectors and IDs
        """
        if self.use_kmeans:
            # Use FAISS k-means clustering to partition the vectors
            import faiss
            import numpy as np

            # Convert tensors to numpy arrays for FAISS
            x = vectors.numpy()
            d = x.shape[1]

            # Run k-means clustering with 20 iterations
            kmeans = faiss.Kmeans(d, self.num_partitions, niter=20)
            kmeans.train(x)

            # Get cluster assignments by finding nearest centroid for each vector
            centroids = kmeans.centroids
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            _, assignments = index.search(x, 1)
            assignments = assignments.ravel()

            # Partition vectors and ids by cluster
            partitioned_vectors = []
            partitioned_ids = []

            for i in range(self.num_partitions):
                mask = assignments == i
                partitioned_vectors.append(vectors[mask])
                partitioned_ids.append(ids[mask])

            return partitioned_vectors, partitioned_ids
        else:
            # Original even splitting logic
            total_size = vectors.size(0)
            base_size = total_size // self.num_partitions
            remainder = total_size % self.num_partitions

            partitioned_vectors = []
            partitioned_ids = []

            start_idx = 0
            for i in range(self.num_partitions):
                # Calculate size for this partition
                partition_size = base_size + (1 if i < remainder else 0)

                # Slice vectors and ids
                end_idx = start_idx + partition_size
                partitioned_vectors.append(vectors[start_idx:end_idx])
                partitioned_ids.append(ids[start_idx:end_idx])

                start_idx = end_idx

            return partitioned_vectors, partitioned_ids

    def build(self, vectors: torch.Tensor, ids: torch.Tensor):
        """
        Build the index on all servers. Each server gets a full copy of the index.

        Args:
            vectors: Tensor of vectors to index
            ids: Tensor of vector IDs
            build_params: Parameters for building the index
        """
        if vectors.size(0) != ids.size(0):
            raise ValueError("Number of vectors must match number of IDs")

        self.partition_to_server_map = defaultdict(list)

        partitioned_vectors, partitioned_ids = self._prepartition_vectors(vectors, ids)

        assert (
            len(partitioned_vectors) == self.num_partitions
        ), "Number of partitioned vectors must match number of partitions"

        # with ThreadPoolExecutor(max_workers=len(self.server_addresses)) as executor:
        #     executor.map(f, range(len(self.server_addresses)))

        print("hello")
        n_servers = len(self.server_addresses)
        futures = []
        with ThreadPoolExecutor(max_workers=n_servers) as executor:
            for i in range(n_servers):
                print("Submitting", i)
                partition_idx = i % self.num_partitions
                v = partitioned_vectors[partition_idx]
                ids = partitioned_ids[partition_idx]
                build_params = self.build_params[i]
                future = executor.submit(self._f, i, v, ids, build_params, partition_idx)
                futures.append(future)

            # Collect results as they complete
            results = [future.result() for future in futures]

        # # Create build_params for each server
        # for i in range(len(self.server_addresses)):
        #     # Build the index on each server
        #     partition_idx = i % self.num_partitions
        #     self.indices[i].build(partitioned_vectors[partition_idx], partitioned_ids[partition_idx], self.build_params[i])
        #     self.partition_to_server_map[partition_idx].append(self.server_addresses[i])

        print("Partition to server map:")
        print(self.partition_to_server_map)

    def _f(self, i, v, ids, build_params, partition_idx):
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Started building index", i)
        self.indices[i].build(v, ids, build_params)
        self.partition_to_server_map[partition_idx].append(self.server_addresses[i])
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Finished building index", i)

    def get_index_and_params(self, server_address: str):
        """
        Get the index and params for a given server address.
        """
        for i in range(len(self.server_addresses)):
            if self.server_addresses[i] == server_address:
                return self.indices[i], self.build_params[i], self.search_params[i]

    def _search_single_server(self, server_idx: int, queries: torch.Tensor, ts) -> torch.Tensor:
        print("took to start", server_idx, time.time() - ts)
        print("!END start job #", server_idx, time.time())
        print("!START run job #", server_idx, time.time())
        """Helper method to perform search on a single server."""
        start = time.perf_counter()
        r = self.indices[server_idx].search(queries, self.search_params[server_idx])
        end = time.perf_counter()
        print("!END run job #", server_idx, time.time())
        self.stats["num_queries"] += 1
        self.stats["time_queries"] += end - start
        return r

    def _search_single_server_dist(self, server_address: str, queries: torch.Tensor, ts) -> torch.Tensor:
        """Helper method to perform search on a single server."""
        print("took to start", server_address, time.time() - ts)
        print("!END start job #", server_address, time.time())
        print("!START run job #", server_address, time.time())
        start = time.perf_counter()
        index, _, search_params = self.get_index_and_params(server_address)
        rr = index.search(queries, search_params)
        r = (rr.ids, rr.distances)
        end = time.perf_counter()
        print("!END run job #", server_address, time.time())
        self.stats["num_queries"] += 1
        self.stats["time_queries"] += end - start
        return r

    def search(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Distribute queries across servers in parallel and merge results.

        Args:
            queries: Tensor of query vectors

        Returns:
            Search results from all servers merged and sorted
        """
        print("!START search", time.time())

        # Distribute queries across servers
        n_servers = len(self.server_addresses)
        n_queries = queries.size(0)

        # Calculate how many queries each server should handle
        queries_per_server = n_queries // n_servers
        remainder = n_queries % n_servers

        # Split queries among servers
        start_idx = 0
        futures = []

        print("!START thread pool create", time.time())
        with ThreadPoolExecutor(max_workers=n_servers) as executor:
            print("!END thread pool create", time.time())
            for i in range(n_servers):
                # Calculate number of queries for this server
                n_queries_for_server = queries_per_server + (1 if i < remainder else 0)
                if n_queries_for_server == 0:
                    continue

                # Get queries for this server
                end_idx = start_idx + n_queries_for_server
                server_queries = queries[start_idx:end_idx]

                # Submit search task to thread pool
                print("!START submitting job #", i, time.time())
                print("!START start job #", i, time.time())
                future = executor.submit(self._search_single_server, i, server_queries, time.time())
                print("!END submitting job #", i, time.time())
                futures.append(future)
                start_idx = end_idx

            # Collect results as they complete
            results = [future.result() for future in futures]

        # Merge results
        start = time.perf_counter()
        r = self._merge_search_results(results)
        end = time.perf_counter()
        self.stats["num_merges"] += 1
        self.stats["time_merges"] += end - start
        print("!END search", time.time())
        return r

    def search_dist(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Distribute queries across servers in parallel and merge results.

        Args:
            queries: Tensor of query vectors

        Returns:
            Search results from all servers merged and sorted
        """
        print("!START search", time.time())
        # Distribute queries across servers
        n_servers = len(self.server_addresses)
        num_replicas = n_servers // self.num_partitions

        # Split queries among servers
        self.results_list = []  # Reset results list

        print("!START thread pool create", time.time())
        with ThreadPoolExecutor(max_workers=n_servers) as executor:
            print("!END thread pool create", time.time())
            # Calculate base batch size and remainder
            num_queries = len(queries)
            base_batch_size = num_queries // num_replicas
            remainder = num_queries % num_replicas

            for i in range(num_replicas):
                futures = []
                # Calculate batch size for this partition
                batch_size = base_batch_size + (1 if i < remainder else 0)
                if batch_size == 0:
                    continue

                # Get queries for this partition
                start_idx = i * base_batch_size + min(i, remainder)
                end_idx = start_idx + batch_size
                queries_for_partition_i = queries[start_idx:end_idx]

                # Submit to all servers handling this partition
                servers_to_submit = [value[i] for value in self.partition_to_server_map.values()]
                for server in servers_to_submit:
                    print("!START submitting job #", server, time.time())
                    print("!START start job #", server, time.time())
                    future = executor.submit(
                        self._search_single_server_dist, server, queries_for_partition_i, time.time()
                    )
                    print("!END submitting job #", server, time.time())
                    futures.append(future)
                print("!START collecting results", time.time())
                results = [future.result() for future in futures]
                print("!END collecting results", time.time())
                self.results_list.append(results)

        # Merge results
        start = time.perf_counter()
        r = self._merge_search_results_dist(self.results_list)
        end = time.perf_counter()
        self.stats["num_merges"] += 1
        self.stats["time_merges"] += end - start
        print("!END search", time.time())
        return r

    def search_sync(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Synchronous wrapper for the async search method.
        """
        return asyncio.run(self.search(queries))

    def add(self, vectors: torch.Tensor, ids: torch.Tensor):
        """
        Add vectors to all servers' indices.

        Args:
            vectors: Tensor of vectors to add
            ids: Tensor of vector IDs
        """
        for index in self.indices:
            index.add(vectors, ids)

    def remove(self, ids: torch.Tensor):
        """
        Remove vectors from all servers' indices.

        Args:
            ids: Tensor of vector IDs to remove
        """
        for index in self.indices:
            index.remove(ids)

    def _merge_search_results(self, results: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge search results from multiple servers.
        Since each server handled a different subset of queries, we just concatenate the results.

        Args:
            results: A list of type distributedwrapper.rwrapper.Local, we can obtain the tensors from the ids

        Returns:
            Concatenated search results
        """
        #
        print("!START collecting ids", time.time())
        ids = [result.ids for result in results]
        print("!END collecting ids", time.time())
        print("!START processing results", time.time())
        ids = torch.cat(ids, dim=0)
        print("!END processing results", time.time())
        return ids

    def _merge_search_results_dist(self, results_list: List[List]) -> torch.Tensor:
        """
        Merge search results from multiple servers.
        Since each server handled a different subset of queries, we just concatenate the results.

        Args:
            results: A list of type distributedwrapper.rwrapper.Local, we can obtain the tensors from the ids

        Returns:
            Concatenated search results of shape (num_queries, k)
        """
        full_ids = []
        for i in range(len(results_list)):
            # Get all IDs and distances for this partition
            print("!START ids distances", time.time())
            ids = [result[0] for result in results_list[i]]
            print("!END ids distances", time.time())
            print("!START collecting distances", time.time())
            distances = [result[1] for result in results_list[i]]
            print("!END collecting distances", time.time())

            print("!START processing results #", i, time.time())
            # Concatenate along the k dimension (dim=1)
            ids = torch.cat(ids, dim=1)  # shape: (num_queries, total_k)
            distances = torch.cat(distances, dim=1)  # shape: (num_queries, total_k)

            # Sort by distances and get top k
            sorted_indices = torch.argsort(distances, dim=1)
            sorted_ids = torch.gather(ids, 1, sorted_indices)

            # Take top k results
            top_k_ids = sorted_ids[:, : self.k]
            full_ids.append(top_k_ids)
            print("!END processing results #", i, time.time())

        # Concatenate results from all partitions
        print("!START finalize results", time.time())
        final_ids = torch.cat(full_ids, dim=0)
        print("!END finalize results", time.time())
        return final_ids

    def calculate_gini_index(self) -> float:
        """
        Calculate the Gini index for the distribution of top k results across servers.
        A Gini index of 0 indicates perfect equality (results evenly distributed),
        while 1 indicates maximum inequality (all results from a single server).

        Returns:
            float: Gini index between 0 and 1
        """
        return 0
        # if not self.top_k_server_counts:
        #     return 0.0
        #
        # # Convert counts to proportions
        # total_results = sum(self.top_k_server_counts.values())
        # if total_results == 0:
        #     return 0.0
        #
        # proportions = [count / total_results for count in self.top_k_server_counts.values()]
        # proportions.sort()
        #
        # # Calculate Gini index
        # n = len(proportions)
        # gini_sum = 0
        # for i in range(n):
        #     for j in range(n):
        #         gini_sum += abs(proportions[i] - proportions[j])
        #
        # gini_index = gini_sum / (2 * n * sum(proportions))
        # return gini_index
