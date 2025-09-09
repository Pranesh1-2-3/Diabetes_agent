"""FastAPI endpoint for RAG-based document search."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

app = FastAPI(
    title="Medical Guidelines Search API",
    description="Search medical guidelines using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configurations
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # now points to Code
INDEX_DIR = ROOT_DIR / "data" / "index"
INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "chunks_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Looking for index at: {INDEX_PATH}")
print(f"Looking for metadata at: {METADATA_PATH}")

# Ensure the paths exist
if not INDEX_PATH.exists():
    raise FileNotFoundError(f"Index file not found at {INDEX_PATH}")
if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

# Load index and metadata
try:
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH) as f:
        chunks_metadata = json.load(f)
except Exception as e:
    print(f"Error loading files: {e}")
    raise

# Initialize the model
model = SentenceTransformer(MODEL_NAME)


async def search_guidelines(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Wrapper function to search medical guidelines.
    """
    search_input = SearchQuery(query=query, top_k=top_k)
    response = await search(search_input)

    results = []
    for r in response.results:
        if isinstance(r, dict):
            results.append({
                "doc_id": r.get("doc_id"),
                "page_num": r.get("page_num"),
                "text": r.get("text")
            })
        else:  # object with attributes
            results.append({
                "doc_id": getattr(r, "doc_id", None),
                "page_num": getattr(r, "page_num", None),
                "text": getattr(r, "text", None)
            })

    return results

class SearchQuery(BaseModel):
    """Input schema for search endpoint."""
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    """Schema for a single search result."""
    doc_id: str
    page_num: int
    chunk_idx: int
    text: str
    score: float

class SearchResponse(BaseModel):
    """Output schema for search endpoint."""
    query: str
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Search for relevant chunks given a query."""
    try:
        # Generate query embedding
        query_embedding = model.encode([query.query])
        
        # Search in FAISS index
        distances, indices = index.search(
            query_embedding.astype('float32'), 
            query.top_k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk_data = chunks_metadata[idx].copy()
            results.append(SearchResult(
                doc_id=chunk_data['doc_id'],
                page_num=chunk_data['page_num'],
                chunk_idx=chunk_data['chunk_idx'],
                text=chunk_data['text'],
                score=float(distance)
            ))
        
        return SearchResponse(
            query=query.query,
            results=results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
