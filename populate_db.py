import os
import glob
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database import Identity, DATABASE_URL  # Import from our database.py
from deepface import DeepFace
from tqdm import tqdm # A nice progress bar

# --- Database Connection ---
# We re-create the engine and session here for this script
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

# --- Configuration ---
GALLERY_PATH = "gallery"
# Ensure we use the exact models specified in the assignment
# We'll use RetinaFace to detect and align the face
# We'll use ArcFace to get the final embedding
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"

def generate_and_store_embeddings():
    print(f"Starting embedding generation for all images in '{GALLERY_PATH}'...")
    print(f"Using Model: {MODEL_NAME} and Detector: {DETECTOR_BACKEND}")

    # Use glob to find all .jpg images recursively
    # This finds 'gallery/person_a/1.jpg', 'gallery/person_a/2.jpg', etc.
    image_paths = glob.glob(os.path.join(GALLERY_PATH, "*", "*.jpg"))
    
    if not image_paths:
        print(f"Error: No images found in '{GALLERY_PATH}'.")
        print("Please add images in the format: gallery/<person_name>/image.jpg")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # 1. Check if this image is already in the DB
            exists = db.query(Identity).filter(Identity.image_path == img_path).first()
            if exists:
                # print(f"Skipping {img_path}, already in database.")
                continue

            # 2. Extract the person's name from the folder path
            # e.g., 'gallery/person_a/1.jpg' -> 'person_a'
            name = os.path.basename(os.path.dirname(img_path))

            # 3. Generate the embedding. This is the core DeepFace magic.
            # It handles detection, alignment, and embedding extraction in one go.
            # The result is a list of embeddings. We take the first one.
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True # Skip images where no face is found
            )
            
            # The 'embedding' is a list of floats.
            embedding = embedding_objs[0]['embedding']

            # 4. Convert the list of floats to numpy array and then to raw bytes (BLOB)
            embedding_bytes = np.array(embedding).astype(np.float32).tobytes()

            # 5. Create the new database record
            new_identity = Identity(
                name=name,
                image_path=img_path,
                embedding=embedding_bytes
            )

            # 6. Add to session and commit
            db.add(new_identity)
            db.commit()

        except ValueError as e:
            # This block catches errors from DeepFace, e.g., "Face could not be detected"
            print(f"\nSkipping {img_path}: {e}")
            db.rollback() # Rollback the failed transaction
        except Exception as e:
            print(f"\nAn error occurred with {img_path}: {e}")
            db.rollback()

    print("\nEmbedding generation complete.")
    total_entries = db.query(Identity).count()
    print(f"Database now contains {total_entries} entries.")

if __name__ == "__main__":
    generate_and_store_embeddings()
    db.close()