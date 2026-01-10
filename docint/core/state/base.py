from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# --- Session persistence (ORM) ---
Base = declarative_base()


# --- Session maker ---
def _make_session_maker(db_url: str) -> sessionmaker:
    """
    Creates a new SQLAlchemy session maker.

    Args:
        db_url (str): The database URL.

    Returns:
        sessionmaker: The SQLAlchemy session maker.
    """
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
