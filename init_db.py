from app.models import Base, User
from sqlalchemy import create_engine
import os

# Get the absolute path to the database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'app', 'site.db')

# Remove existing database if it exists
if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)

# Create new database
engine = create_engine(f'sqlite:///{DATABASE_PATH}')
Base.metadata.create_all(engine)

print(f"Database initialized at {DATABASE_PATH}") 