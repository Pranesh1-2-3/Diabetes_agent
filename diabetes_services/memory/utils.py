"""
Memory utilities for storing and retrieving case histories.
Includes anonymization, embedding generation, and metadata management.
"""

import json
from pathlib import Path
import numpy as np
import faiss
from datetime import datetime
from typing import Dict, List, Optional, Union
import hashlib

# Constants
MEMORY_DIR = Path(__file__).parent.parent.parent / "data" / "memory"
FAISS_INDEX_PATH = MEMORY_DIR / "case_memory.index"
METADATA_PATH = MEMORY_DIR / "case_metadata.json"

class CaseMemoryManager:
    def __init__(self, dimension: int = 768):  # Default for sentence-transformers
        """Initialize the case memory system with FAISS index."""
        self.dimension = dimension
        self.load_or_create_index()
        self.load_metadata()

    def load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        if FAISS_INDEX_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def load_metadata(self):
        """Load case metadata from JSON file."""
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self.save_metadata()

    def save_metadata(self):
        """Save metadata to JSON file."""
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(METADATA_PATH, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def anonymize_case(self, case_data: Dict) -> Dict:
        """Create anonymized version of case data."""
        anon_data = case_data.copy()
        
        # Generate consistent anonymous ID
        case_str = json.dumps(case_data, sort_keys=True)
        case_hash = hashlib.sha256(case_str.encode()).hexdigest()[:8]
        anon_data["case_id"] = f"CASE_{case_hash}"
        
        # Remove or mask sensitive fields
        sensitive_fields = ["name", "patient_id", "email", "phone", "address"]
        for field in sensitive_fields:
            if field in anon_data:
                del anon_data[field]
        
        return anon_data

    def upsert_case(self, 
                    case_data: Dict,
                    embedding: np.ndarray,
                    model_versions: Dict[str, str],
                    outcome: Optional[str] = None,
                    synopsis: Optional[str] = None) -> str:
        """Store or update a case in memory."""
        # Anonymize case data
        anon_data = self.anonymize_case(case_data)
        case_id = anon_data["case_id"]
        
        # Prepare metadata
        metadata = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            "model_versions": model_versions,
            "outcome": outcome,
            "synopsis": synopsis,
            "data": anon_data
        }
        
        # Update FAISS index
        embedding = embedding.astype(np.float32).reshape(1, -1)
        if case_id in self.metadata:
            # Update existing case
            idx = self.metadata[case_id]["index"]
            self.index.remove_ids(np.array([idx]))
        else:
            # Add new case
            idx = self.index.ntotal
        
        self.index.add(embedding)
        metadata["index"] = idx
        
        # Update metadata
        self.metadata[case_id] = metadata
        self.save_metadata()
        
        return case_id

    def query_similar_cases(
        self,
        embedding: np.ndarray,
        k: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """Query for similar cases using embedding (with FAISS dimension validation)."""
        import numpy as np

        # --- 🧠 Ensure embedding is float32 and 2D ---
        if embedding is None:
            raise ValueError("❌ Received None as embedding input.")

        embedding = np.array(embedding, dtype=np.float32)
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        # --- 🔍 Validate FAISS index dimension ---
        expected_dim = getattr(self.index, "d", None)
        actual_dim = embedding.shape[1]

        if expected_dim is None:
            raise ValueError("❌ FAISS index not initialized or missing dimension info (index.d).")

        if actual_dim != expected_dim:
            raise ValueError(
                f"❌ Embedding dimension mismatch:\n"
                f"   → FAISS index dimension: {expected_dim}\n"
                f"   → Embedding dimension: {actual_dim}\n"
                f"⚠️ Rebuild the FAISS index using embeddings from the same model (e.g., 'all-MiniLM-L6-v2' = 384)."
            )

        # --- ✅ Perform FAISS search ---
        distances, indices = self.index.search(embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Skip invalid or low-similarity results
            if idx == -1 or dist > (1 - min_similarity):
                continue

            # Find metadata for this index
            for case_id, meta in self.metadata.items():
                if meta["index"] == idx:
                    result = {
                        "case_id": case_id,
                        "similarity": 1 - (dist / 2),  # Convert L2 distance to similarity
                        "timestamp": meta.get("timestamp"),
                        "outcome": meta.get("outcome"),
                        "synopsis": meta.get("synopsis"),
                        "data": meta.get("data"),
                    }
                    results.append(result)
                    break

        return results
