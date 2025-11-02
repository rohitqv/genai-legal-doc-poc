"""
Embedding and FAISS index management module
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import torch

from ..utils.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, FAISS_INDEX_PATH
from ..utils.logger import logger


class EmbeddingManager:
    """Class for managing document embeddings and FAISS index"""
    
    def __init__(self, model_name: str = None, dimension: int = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.dimension = dimension or EMBEDDING_DIMENSION
        self.model = None
        self.index = None
        self.metadata = []
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return np.array([])
        
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            logger.info(f"Created embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def create_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            use_gpu: Whether to use GPU (if available)
            
        Returns:
            FAISS index
        """
        if embeddings.size == 0:
            raise ValueError("Cannot create index from empty embeddings")
        
        logger.info(f"Creating FAISS index for {len(embeddings)} embeddings...")
        
        # Use Inner Product (IP) for normalized embeddings (equivalent to cosine similarity)
        # For non-normalized, use L2 distance
        index = faiss.IndexFlatIP(self.dimension)
        
        if use_gpu and torch.cuda.is_available():
            logger.info("Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            logger.info("Using CPU for FAISS")
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        return index
    
    def add_to_index(
        self,
        texts: List[str],
        metadata: List[Dict],
        index: Optional[faiss.Index] = None
    ) -> faiss.Index:
        """
        Add new texts to existing index or create new one
        
        Args:
            texts: List of text strings
            metadata: List of metadata dictionaries (one per text)
            index: Existing FAISS index (optional)
            
        Returns:
            Updated FAISS index
        """
        if len(texts) != len(metadata):
            raise ValueError("Texts and metadata must have same length")
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Create or update index
        if index is None:
            index = self.create_faiss_index(embeddings)
            self.metadata = metadata.copy()
        else:
            index.add(embeddings.astype('float32'))
            self.metadata.extend(metadata)
        
        logger.info(f"Added {len(texts)} items to index. Total: {index.ntotal}")
        return index
    
    def save_index(self, index: faiss.Index, metadata: List[Dict], file_path: str = None):
        """
        Save FAISS index and metadata to disk
        
        Args:
            index: FAISS index
            metadata: List of metadata dictionaries
            file_path: Path to save index
        """
        file_path = file_path or FAISS_INDEX_PATH
        
        # Create directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert GPU index to CPU if needed
        if isinstance(index, faiss.GpuIndex):
            logger.info("Converting GPU index to CPU for saving...")
            index = faiss.index_gpu_to_cpu(index)
        
        # Save index
        faiss.write_index(index, file_path)
        logger.info(f"Saved FAISS index to {file_path}")
        
        # Save metadata
        metadata_path = file_path.replace('.index', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_index(self, file_path: str = None) -> tuple:
        """
        Load FAISS index and metadata from disk
        
        Args:
            file_path: Path to index file
            
        Returns:
            Tuple of (index, metadata)
        """
        file_path = file_path or FAISS_INDEX_PATH
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Index file not found: {file_path}")
        
        # Load index
        index = faiss.read_index(file_path)
        logger.info(f"Loaded FAISS index from {file_path} with {index.ntotal} vectors")
        
        # Load metadata
        metadata_path = file_path.replace('.index', '_metadata.pkl')
        if Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded metadata with {len(metadata)} entries")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            metadata = []
        
        self.index = index
        self.metadata = metadata
        
        return index, metadata
    
    def search(
        self,
        query: str,
        index: faiss.Index,
        metadata: List[Dict],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar texts in index
        
        Args:
            query: Query text
            index: FAISS index
            metadata: Metadata list
            top_k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            List of results with text, score, and metadata
        """
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            
            result = {
                "index": int(idx),
                "score": float(score),
                "metadata": metadata[idx] if idx < len(metadata) else {}
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} results for query")
        return results

