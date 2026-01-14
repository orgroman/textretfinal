"""
Build a Faiss binary index for the entire Robust04 corpus using binarized BGE embeddings.

This creates a new retrieval method based on Hamming distance similarity
over binary codes derived from dense embeddings.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss

os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false")


def load_faiss_embeddings_and_docids(
    faiss_index_name: str = "beir-v1.0.0-robust04.bge-base-en-v1.5",
    query_encoder: str = "BAAI/bge-base-en-v1.5",
) -> Tuple[np.ndarray, List[str], "FaissSearcher"]:
    """Load precomputed embeddings from Pyserini Faiss index."""
    from pyserini.search.faiss import FaissSearcher

    searcher = FaissSearcher.from_prebuilt_index(faiss_index_name, query_encoder=query_encoder)
    num_docs = searcher.num_docs
    dim = searcher.index.d
    docids = list(searcher.docids)

    print(f"Reconstructing {num_docs} vectors from Faiss index...")
    all_embs = np.zeros((num_docs, dim), dtype=np.float32)
    for i in range(num_docs):
        all_embs[i] = searcher.index.reconstruct(i)
        if (i + 1) % 100000 == 0:
            print(f"  {i+1}/{num_docs} vectors loaded...")

    return all_embs, docids, searcher


def binarize_embeddings(
    embeddings: np.ndarray,
    threshold_vec: np.ndarray,
) -> np.ndarray:
    """
    Binarize embeddings using per-dimension thresholds.
    
    Returns packed uint8 array for Faiss binary index.
    Faiss binary indexes expect d bits packed into d/8 bytes.
    """
    # Binarize: 1 if >= threshold, 0 otherwise
    binary_codes = (embeddings >= threshold_vec).astype(np.uint8)
    
    # Pack bits into bytes (Faiss expects uint8 with 8 bits per byte)
    # For 768 dimensions, we get 768/8 = 96 bytes per vector
    n_docs, n_dims = binary_codes.shape
    assert n_dims % 8 == 0, f"Dimensions must be multiple of 8, got {n_dims}"
    
    n_bytes = n_dims // 8
    packed = np.packbits(binary_codes, axis=1)
    
    return packed  # Shape: (n_docs, n_bytes)


def compute_threshold_vector(
    embeddings: np.ndarray,
    mode: str,
    threshold_path: Optional[str] = None,
    quantile: Optional[float] = None,
) -> np.ndarray:
    if mode == "file":
        if not threshold_path:
            raise ValueError("threshold_path is required when mode='file'")
        vec = np.load(threshold_path)
        return vec.astype(np.float32)

    dim = int(embeddings.shape[1])
    if mode == "mean":
        return embeddings.mean(axis=0).astype(np.float32)
    if mode == "median":
        return np.median(embeddings, axis=0).astype(np.float32)
    if mode == "zero":
        return np.zeros(dim, dtype=np.float32)
    if mode == "quantile":
        if quantile is None:
            raise ValueError("quantile is required when mode='quantile'")
        return np.quantile(embeddings, float(quantile), axis=0).astype(np.float32)

    raise ValueError(f"Unknown threshold mode: {mode}")


def load_packed_codes_from_index(index_path: Path) -> np.ndarray:
    index = faiss.read_index_binary(str(index_path))
    if not hasattr(index, "xb"):
        raise ValueError(
            "Source binary index does not expose raw codes via .xb; expected IndexBinaryFlat. "
            "Rebuild from float embeddings instead."
        )

    n_bits = int(index.d)
    if n_bits % 8 != 0:
        raise ValueError(f"Index bits must be multiple of 8, got {n_bits}")
    code_size = n_bits // 8
    ntotal = int(index.ntotal)

    xb = faiss.vector_to_array(index.xb)
    expected = ntotal * code_size
    if int(xb.size) != int(expected):
        raise ValueError(f"Unexpected xb size: got {xb.size}, expected {expected}")
    return xb.reshape(ntotal, code_size)


def build_binary_index(
    packed_codes: np.ndarray,
    index_type: str = "flat",
    nlist: int = 100,
    nprobe: int = 4,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 256,
    hash_bits: int = 64,
    hash_nhash: int = 4,
    hash_nflip: int = 0,
) -> faiss.IndexBinary:
    """
    Build a Faiss binary index from packed binary codes.
    
    Args:
        packed_codes: Shape (n_docs, n_bytes) uint8 array
        index_type: "flat" for exact search, "ivf" for approximate
        nlist: Number of clusters for IVF index
    
    Returns:
        Faiss binary index
    """
    n_docs, n_bytes = packed_codes.shape
    n_bits = n_bytes * 8
    
    if index_type == "flat":
        index = faiss.IndexBinaryFlat(n_bits)
    elif index_type == "ivf":
        quantizer = faiss.IndexBinaryFlat(n_bits)
        index = faiss.IndexBinaryIVF(quantizer, n_bits, nlist)
        # Train the index
        print(f"Training IVF index with {nlist} clusters...")
        index.train(packed_codes)
        index.nprobe = int(nprobe)
    elif index_type == "hnsw":
        index = faiss.IndexBinaryHNSW(n_bits, int(hnsw_m))
        if hasattr(index, "hnsw"):
            if hasattr(index.hnsw, "efConstruction"):
                index.hnsw.efConstruction = int(hnsw_ef_construction)
            if hasattr(index.hnsw, "efSearch"):
                index.hnsw.efSearch = int(hnsw_ef_search)
    elif index_type == "hash":
        hb = int(hash_bits)
        if hb <= 0 or hb > n_bits:
            raise ValueError(f"hash_bits must be in [1, {n_bits}], got {hb}")
        index = faiss.IndexBinaryHash(n_bits, hb)
        if hasattr(index, "nflip"):
            index.nflip = int(hash_nflip)
    elif index_type == "multihash":
        hb = int(hash_bits)
        nh = int(hash_nhash)
        if hb <= 0 or hb > n_bits:
            raise ValueError(f"hash_bits must be in [1, {n_bits}], got {hb}")
        if nh <= 0:
            raise ValueError(f"hash_nhash must be > 0, got {nh}")
        if nh * hb > n_bits:
            raise ValueError(f"hash_nhash * hash_bits must be <= {n_bits}, got {nh * hb}")
        index = faiss.IndexBinaryMultiHash(n_bits, nh, hb)
        if hasattr(index, "nflip"):
            index.nflip = int(hash_nflip)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    print(f"Adding {n_docs} vectors to index...")
    index.add(packed_codes)
    
    return index


def search_binary_index(
    index: faiss.IndexBinary,
    query_codes: np.ndarray,
    k: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the binary index.
    
    Args:
        index: Faiss binary index
        query_codes: Packed query codes, shape (n_queries, n_bytes)
        k: Number of results per query
    
    Returns:
        (distances, indices) - Hamming distances and doc indices
    """
    distances, indices = index.search(query_codes, k)
    return distances, indices


class BinaryRetriever:
    """Wrapper for binary index retrieval compatible with fusion pipeline."""
    
    def __init__(
        self,
        index_path: str,
        docids_path: str,
        threshold_path: str,
        query_encoder: str = "BAAI/bge-base-en-v1.5",
    ):
        self.index = faiss.read_index_binary(index_path)
        self.docids = np.load(docids_path, allow_pickle=True).tolist()
        self.threshold_vec = np.load(threshold_path)
        
        # Load query encoder
        from pyserini.search.faiss import FaissSearcher
        self.faiss_searcher = FaissSearcher.from_prebuilt_index(
            "beir-v1.0.0-robust04.bge-base-en-v1.5",
            query_encoder=query_encoder,
        )
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode and binarize a query."""
        q_emb = np.array(self.faiss_searcher.query_encoder.encode(query), dtype=np.float32)
        q_binary = (q_emb >= self.threshold_vec).astype(np.uint8)
        q_packed = np.packbits(q_binary).reshape(1, -1)
        return q_packed
    
    def search(self, query: str, k: int = 1000) -> List[Tuple[str, float]]:
        """
        Search for a query and return (docid, score) pairs.
        
        Score is converted from Hamming distance to similarity:
        similarity = 1 - (hamming_dist / n_bits)
        """
        q_packed = self.encode_query(query)
        distances, indices = self.index.search(q_packed, k)
        
        n_bits = self.index.d
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            docid = self.docids[idx]
            # Convert Hamming distance to similarity
            similarity = 1.0 - (dist / n_bits)
            results.append((docid, similarity))
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Build binary Faiss index for Robust04")
    parser.add_argument("--faiss-index", default="beir-v1.0.0-robust04.bge-base-en-v1.5")
    parser.add_argument("--query-encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reuse-from", default=None, help="Reuse packed codes from an existing binary index dir")
    parser.add_argument("--threshold", default=None, help="Path to threshold .npy file (default: use mean)")
    parser.add_argument(
        "--threshold-mode",
        default="mean",
        choices=["mean", "median", "zero", "quantile", "file"],
    )
    parser.add_argument("--threshold-quantile", type=float, default=None)
    parser.add_argument("--index-type", default="flat", choices=["flat", "ivf", "hnsw", "hash", "multihash"])
    parser.add_argument("--nlist", type=int, default=100, help="Number of clusters for IVF")
    parser.add_argument("--nprobe", type=int, default=4)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=256)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--hash-nhash", type=int, default=4)
    parser.add_argument("--hash-nflip", type=int, default=0)
    parser.add_argument("--output-dir", default="binary_index")
    parser.add_argument("--test-queries", default="Files-20260104/queriesROBUST.txt")
    parser.add_argument("--test-k", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    
    reuse_dir = args.reuse_from
    reused_codes = False
    if reuse_dir:
        reuse_dir = str(reuse_dir)
        reuse_path = Path(reuse_dir)
        src_index = reuse_path / "binary_flat.faiss"
        if not src_index.exists():
            candidates = sorted(reuse_path.glob("binary_*.faiss"))
            if len(candidates) == 1:
                src_index = candidates[0]
            else:
                raise FileNotFoundError(
                    f"Could not find binary_flat.faiss in {reuse_path} and could not auto-pick a unique binary_*.faiss"
                )

        print(f"Reusing packed codes from {src_index}")
        packed_codes = load_packed_codes_from_index(src_index)
        docids = np.load(reuse_path / "docids.npy", allow_pickle=True).tolist()
        threshold_vec = np.load(reuse_path / "threshold.npy")
        reused_codes = True
        print(f"Loaded packed codes: {packed_codes.shape} (dtype: {packed_codes.dtype})")
        print(f"Loaded docids: {len(docids)}")
        print(f"Loaded threshold vector: {threshold_vec.shape}")

        faiss_searcher = None
    else:
        # Load embeddings
        print("Loading embeddings from Faiss index...")
        embeddings, docids, faiss_searcher = load_faiss_embeddings_and_docids(
            faiss_index_name=args.faiss_index,
            query_encoder=args.query_encoder,
        )
        print(f"Loaded {len(docids)} embeddings, shape: {embeddings.shape}")

        # Determine threshold vector
        threshold_mode = str(args.threshold_mode)
        threshold_path = args.threshold
        if threshold_path and threshold_mode != "file":
            threshold_mode = "file"
        if threshold_mode == "file":
            print(f"Loading threshold from {threshold_path}")
        else:
            print(f"Computing threshold: mode={threshold_mode}")
        threshold_vec = compute_threshold_vector(
            embeddings,
            mode=threshold_mode,
            threshold_path=threshold_path,
            quantile=args.threshold_quantile,
        )

        print(f"Threshold vector shape: {threshold_vec.shape}")

        # Binarize embeddings
        print("Binarizing embeddings...")
        packed_codes = binarize_embeddings(embeddings, threshold_vec)
        print(f"Packed codes shape: {packed_codes.shape} (dtype: {packed_codes.dtype})")
        print(f"Storage: {packed_codes.nbytes / 1e6:.1f} MB (vs {embeddings.nbytes / 1e6:.1f} MB for float32)")
    
    # Build index
    print(f"Building {args.index_type} binary index...")
    index = build_binary_index(
        packed_codes,
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        hash_bits=args.hash_bits,
        hash_nhash=args.hash_nhash,
        hash_nflip=args.hash_nflip,
    )
    print(f"Index built: {index.ntotal} vectors, {index.d} bits")
    
    # Save index and metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / f"binary_{args.index_type}.faiss"
    docids_path = output_dir / "docids.npy"
    threshold_path = output_dir / "threshold.npy"
    
    print(f"Saving index to {index_path}")
    faiss.write_index_binary(index, str(index_path))
    np.save(docids_path, np.array(docids, dtype=object))
    np.save(threshold_path, threshold_vec)
    
    # Quick test search
    if args.test_queries:
        if faiss_searcher is None:
            from pyserini.search.faiss import FaissSearcher

            faiss_searcher = FaissSearcher.from_prebuilt_index(args.faiss_index, query_encoder=args.query_encoder)
        print(f"\nTesting with sample queries from {args.test_queries}...")
        queries = {}
        with open(args.test_queries) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                qid, query = line.split("\t", 1)
                queries[qid] = query
        
        # Test first 3 queries
        test_qids = list(queries.keys())[:3]
        for qid in test_qids:
            query = queries[qid]
            q_emb = np.array(faiss_searcher.query_encoder.encode(query), dtype=np.float32)
            q_binary = (q_emb >= threshold_vec).astype(np.uint8)
            q_packed = np.packbits(q_binary).reshape(1, -1)
            
            distances, indices = index.search(q_packed, args.test_k)
            
            print(f"\nQuery {qid}: {query[:50]}...")
            print(f"  Top {args.test_k} results (Hamming distance):")
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    print(f"    {docids[idx]}: dist={dist}")
    
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"\nFiles saved:")
    print(f"  Index: {index_path}")
    print(f"  Docids: {docids_path}")
    print(f"  Threshold: {threshold_path}")


if __name__ == "__main__":
    main()
