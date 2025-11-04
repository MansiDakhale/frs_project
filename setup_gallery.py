# setup_gallery_v2.py
import os
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
GALLERY_DIR = "gallery"
# We'll get people with at least 5 images to have some variety
MIN_FACES_PER_PERSON = 5 
# ---------------------

print("Downloading LFW dataset via scikit-learn (this may take a moment)...")

# 'color=True' gets the 3-channel RGB images
# 'funneled=True' is the standard, aligned version which is good for recognition
lfw_people = fetch_lfw_people(
    min_faces_per_person=MIN_FACES_PER_PERSON, 
    color=True, 
    funneled=True
)

# lfw_people.images is a (n_samples, height, width, 3) numpy array
# lfw_people.target is a (n_samples,) array of integer IDs
# lfw_people.target_names is the list of names matching the IDs

print(f"\nDownloaded {len(lfw_people.images)} images for {len(lfw_people.target_names)} people.")
print(f"Saving images to '{GALLERY_DIR}' directory...")

os.makedirs(GALLERY_DIR, exist_ok=True)

# Keep track of image counts per person to name them 1.jpg, 2.jpg, etc.
counts = {}

# Loop over all downloaded images and targets
for i in tqdm(range(len(lfw_people.images)), desc="Saving images"):
    try:
        # Get the name and ID
        person_id = lfw_people.target[i]
        person_name = lfw_people.target_names[i]
        
        # Fix whitespace in names for folder paths
        person_name = person_name.replace(" ", "_")
        
        # Create a directory for the person
        person_dir = os.path.join(GALLERY_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Get the image data (it's in 0-1 float format)
        image_data = lfw_people.images[i]
        
        # Increment count for this person
        if person_name not in counts:
            counts[person_name] = 0
        counts[person_name] += 1
        
        # Create a unique filename
        file_name = f"{counts[person_name]}.jpg"
        file_path = os.path.join(person_dir, file_name)
        
        # Save the image
        # plt.imsave handles the float[0,1] -> JPG conversion perfectly
        plt.imsave(file_path, image_data)
        
    except Exception as e:
        print(f"\nError saving image {i}: {e}")

print("\n✅ Gallery setup complete!")
print(f"Found and saved images for: {list(counts.keys())}")
print("\nYou can now run 'python populate_db.py' to fill the database.")