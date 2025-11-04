# database.py
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base

# --- Database Setup ---
DATABASE_URL = "sqlite:///./frs.db"


engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

Base = declarative_base()

# --- Define Our Table ---
class Identity(Base):
    __tablename__ = "identities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    image_path = Column(String, unique=True)
    # We will store the embedding as a BLOB (LargeBinary)
    embedding = Column(LargeBinary)

# --- Function to Create the Database ---
def create_db_and_tables():
    try:
        Base.metadata.create_all(bind=engine)
        print("Database 'frs.db' and 'identities' table created successfully.")
    except Exception as e:
        print(f"An error occurred during table creation: {e}")

# This makes the script runnable from the command line
if __name__ == "__main__":
    create_db_and_tables()