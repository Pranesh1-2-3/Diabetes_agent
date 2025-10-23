"""FastAPI endpoint for medical table parsing."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
from typing import Dict, Any

from ..parser import MedicalTableParser

app = FastAPI(
    title="Medical Table Parser API",
    description="API for parsing medical tables from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize parser
parser = MedicalTableParser(use_advanced_model=False)

@app.post("/parse")
async def parse_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Parse medical table from uploaded image."""
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Parse the image
        result = parser.parse_image(image)
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing image: {str(e)}"
        )