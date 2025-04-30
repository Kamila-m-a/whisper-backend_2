from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
import numpy as np
import os
import tempfile
import gc
import logging
from typing import Dict

# --- Configuration ---
MODEL_NAME = "tiny"  # English-optimized small model

# --- Initialization ---
app = FastAPI(title="Whisper Transcription API")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# --- Model Loading ---
def load_model():
    """Load Whisper model with memory optimizations"""
    logger.info("Loading Whisper model...")
    
    # Pre-initialize NumPy to prevent "Numpy not available" errors
    np.zeros(1)  
    
    model = whisper.load_model(
        MODEL_NAME,
        device="cpu",
        download_root="/tmp/whisper"
    )
    model.eval()
    torch.set_num_threads(1)  # Limit CPU threads
    
    logger.info(f"Loaded {MODEL_NAME} model")
    return model

model = load_model()

# --- Helper Functions ---
def cleanup_temp_files(path: str):
    """Ensure temp files are deleted"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Failed to delete temp file: {e}")

# --- API Endpoints ---
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe audio file using Whisper"""
    temp_path = None
    try:
        # 1. Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(400, "Only audio files are supported")

        # 2. Save to temp file
        _, temp_path = tempfile.mkstemp(suffix=".wav")
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # 3. Transcribe
        logger.info(f"Transcribing {file.filename}...")
        with torch.inference_mode():
            result = model.transcribe(temp_path, fp16=False)

        return {"text": result["text"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        # 4. Cleanup
        if temp_path:
            cleanup_temp_files(temp_path)
        gc.collect()

@app.get("/", response_model=Dict[str, str])
def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "model": MODEL_NAME,
        "message": "Audio splitting should be done client-side"
    }

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    gc.collect()
