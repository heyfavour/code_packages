import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.QT_models import Base
from db.session import engine

def drop_all():
    Base.metadata.drop_all(engine)

def create_all():
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    drop_all()
    create_all()
