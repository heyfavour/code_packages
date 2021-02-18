from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

url = "mysql://root:wzx940516@49.235.242.224/DWDB?charset=utf8"
engine = create_engine(url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine,autocommit=True)
session = SessionLocal()
