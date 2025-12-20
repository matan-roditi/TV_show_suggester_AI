import numpy as np
from usearch.index import Index


class VectorDB:
    """
    A wrapper around the usearch library for efficient vector search.
    
    Provides methods to build an index, search for nearest neighbors,
    and persist the index to disk.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize the VectorDB with a specific vector dimension.
        
        Args:
            dimension: The dimensionality of vectors to be indexed
        """
        self.dimension = dimension
        self.index = Index(ndim=dimension, metric="cosine")
        self.id_to_vector = {}  # Keep track of vectors for validation
    
    def build_index(self, data: dict):
        """
        Build the index from a dictionary of vectors.
        
        Args:
            data: Dictionary where keys are IDs (int) and values are numpy arrays (vectors)
        
        Raises:
            ValueError: If any vector doesn't match the expected dimension
        """
        # Validate all vectors have the correct dimension
        for id_val, vector in data.items():
            if not isinstance(vector, np.ndarray):
                raise ValueError(f"Vector for ID {id_val} is not a numpy array")
            
            if vector.shape[0] != self.dimension:
                raise ValueError(
                    f"Vector for ID {id_val} has shape {vector.shape[0]}, "
                    f"but expected dimension {self.dimension}"
                )
        
        # Add all vectors to the index
        for id_val, vector in data.items():
            # Ensure vector is float32 for usearch
            vector = np.asarray(vector, dtype=np.float32)
            self.index.add(id_val, vector)
            self.id_to_vector[id_val] = vector
    
    def search(self, query_vector: np.ndarray, limit: int = 5) -> list[tuple]:
        """
        Search for the nearest neighbors of the query vector.
        
        Args:
            query_vector: The query vector as a numpy array
            limit: Maximum number of results to return (default 5)
        
        Returns:
            A list of tuples (id, distance) sorted by distance (closest first)
        """
        # Ensure query vector is float32
        query_vector = np.asarray(query_vector, dtype=np.float32)
        
        # Search the index (usearch uses 'count' parameter for number of results)
        matches = self.index.search(query_vector, count=limit)
        
        # Convert usearch results to list of tuples (id, distance)
        # usearch returns a Matches object with keys and distances attributes
        results = list(zip(matches.keys, matches.distances))
        
        return results
    
    def save_index(self, file_path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            file_path: Path where the index will be saved
        """
        self.index.save(file_path)
    
    def load_index(self, file_path: str) -> None:
        """
        Load the index from disk.
        
        Args:
            file_path: Path to the saved index file
        """
        # Create a new index and load from file
        self.index = Index.restore(file_path, view=False)
