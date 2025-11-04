# main.py
import uvicorn
import numpy as np
import cv2
import io
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from sqlalchemy.orm import sessionmaker, Session
from database import Identity, DATABASE_URL, engine, Base  # Import Base
from deepface import DeepFace
from typing import List
from pydantic import BaseModel # Import BaseModel

# --- Configuration ---
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
RECOGNITION_THRESHOLD = 0.68
GALLERY_PATH = "gallery" # Define gallery path for saving images

# --- Database Session ---
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Re-create tables if they don't exist (e.g., first run)
Base.metadata.create_all(bind=engine) 

def get_db():
    """FastAPI Dependency to manage database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- In-Memory Gallery Cache ---
gallery_cache = {
    "names": [],
    "embeddings": []
}

def load_gallery_from_db(db: Session = SessionLocal()):
    """Loads all embeddings from the database into the in-memory cache."""
    print("Loading gallery embeddings from database...")
    gallery_cache["names"] = []
    gallery_cache["embeddings"] = []
    
    identities = db.query(Identity).all()
    
    if not identities:
        print("Warning: No identities found in the database. Gallery is empty.")
        db.close()
        return

    for identity in identities:
        embedding = np.frombuffer(identity.embedding, dtype=np.float32)
        gallery_cache["names"].append(identity.name)
        gallery_cache["embeddings"].append(embedding)
    
    if gallery_cache["embeddings"]:
        gallery_cache["embeddings"] = np.array(gallery_cache["embeddings"])
    
    db.close()
    print(f"Gallery loaded. {len(gallery_cache['names'])} embeddings in memory.")


# --- FastAPI App ---
app = FastAPI(title="Face Recognition Service")

@app.on_event("startup")
def on_startup():
    """This function runs when the FastAPI app starts."""
    load_gallery_from_db()

# --- Helper Function ---
def find_best_match(target_embedding: np.ndarray):
    """Compares a new embedding against the in-memory gallery."""
    if len(gallery_cache["names"]) == 0:
        return "UNKNOWN", 0.0

    target_norm = np.linalg.norm(target_embedding)
    if target_norm == 0: # Avoid division by zero
        return "UNKNOWN", 0.0

    gallery_norm = np.linalg.norm(gallery_cache["embeddings"], axis=1)
    
    # Compute cosine similarity
    similarity_scores = np.dot(target_embedding, gallery_cache["embeddings"].T) / (target_norm * gallery_norm)

    best_match_index = np.argmax(similarity_scores)
    best_score = similarity_scores[best_match_index]
    
    if best_score >= RECOGNITION_THRESHOLD:
        best_match_name = gallery_cache["names"][best_match_index]
        return best_match_name, best_score
    else:
        return "UNKNOWN", best_score

# --- Pydantic Models (Response Schemas) ---
class RecognitionResponse(BaseModel):
    identity: str
    confidence: float
    bounding_box: dict

class IdentityResponse(BaseModel):
    id: int
    name: str
    image_path: str
    
    # Pydantic v2 config to allow ORM models
    class Config:
        from_attributes = True 

class MessageResponse(BaseModel):
    message: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Face Recognition API is running."}

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(file: UploadFile = File(...)):
    """Recognize a face from an uploaded image."""
    if len(gallery_cache["names"]) == 0:
        raise HTTPException(status_code=503, detail="Gallery is empty. Please add identities.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        embedding_objs = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        
        first_face = embedding_objs[0]
        target_embedding = np.array(first_face["embedding"])
        facial_area = first_face["facial_area"]
        
        name, score = find_best_match(target_embedding)
        
        return {
            "identity": name,
            "confidence": float(score),
            "bounding_box": facial_area
        }

    except ValueError as e:
        if "Face could not be detected" in str(e):
            raise HTTPException(status_code=400, detail="No face detected in the image.")
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- NEW ENDPOINTS ---

@app.post("/add_identity", response_model=IdentityResponse)
async def add_identity(
    name: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """Add a new identity to the gallery."""
    
    # 1. Read and save the image file
    person_dir = os.path.join(GALLERY_PATH, name.replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(person_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Generate embedding
        embedding_objs = DeepFace.represent(
            img_path=file_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        embedding = embedding_objs[0]['embedding']
        embedding_bytes = np.array(embedding).astype(np.float32).tobytes()

        # 3. Save to Database
        new_identity = Identity(
            name=name,
            image_path=file_path,
            embedding=embedding_bytes
        )
        db.add(new_identity)
        db.commit()
        db.refresh(new_identity) # Get the new ID from the DB

        # 4. CRITICAL: Update the in-memory cache
        # We append the new data to the live cache
        gallery_cache["names"].append(name)
        new_embeddings = np.array([embedding]) # New embedding as 2D array
        if len(gallery_cache["embeddings"]) == 0:
            gallery_cache["embeddings"] = new_embeddings
        else:
            gallery_cache["embeddings"] = np.concatenate(
                (gallery_cache["embeddings"], new_embeddings)
            )

        print(f"Added new identity: {name}. Cache updated.")
        
        # Return the pydantic model (will be converted to JSON)
        return new_identity 

    except ValueError as e:
        # This error is raised by DeepFace if no face is detected
        if "Face could not be detected" in str(e):
            # Clean up the saved file if no face was found
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"No face detected in image for {name}.")
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/list_identities", response_model=List[str])
def list_identities():
    """List all unique identity names in the gallery."""
    # We read from the live cache, which is much faster than a DB query
    unique_names = sorted(list(set(gallery_cache["names"])))
    return unique_names

# --- How to run this app ---
# In your terminal, run:
# uvicorn main:app --reload