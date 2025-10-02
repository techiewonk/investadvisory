from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.settings import DatabaseType, settings
from memory.mongodb import get_mongo_saver
from memory.postgres import get_postgres_saver, get_postgres_store
from memory.sqlite import get_sqlite_saver, get_sqlite_store


def initialize_database() -> AbstractAsyncContextManager[
    AsyncSqliteSaver | AsyncPostgresSaver | AsyncMongoDBSaver
]:
    """
    Initialize the appropriate database checkpointer based on configuration.
    Returns an initialized AsyncCheckpointer instance.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_saver()
    if settings.DATABASE_TYPE == DatabaseType.MONGO:
        return get_mongo_saver()
    else:  # Default to SQLite
        return get_sqlite_saver()


def initialize_store():
    """
    Initialize the appropriate store based on configuration.
    Returns an async context manager for the initialized store.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_store()
    # TODO: Add Mongo store - https://pypi.org/project/langgraph-store-mongodb/
    else:  # Default to SQLite
        return get_sqlite_store()


def initialize_portfolio_database():
    """
    Initialize the appropriate database connection for portfolio service.
    Returns a database connection that can be used for raw SQL queries.
    """
    # Auto-seed portfolio database on first access
    _auto_seed_portfolio_database()
    
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        from memory.postgres import get_postgres_connection_pool
        return get_postgres_connection_pool()
    elif settings.DATABASE_TYPE == DatabaseType.SQLITE:
        from memory.sqlite import get_sqlite_connection
        return get_sqlite_connection()
    else:
        raise ValueError(f"Unsupported database type for portfolio service: {settings.DATABASE_TYPE}")


def _auto_seed_portfolio_database():
    """Automatically seed portfolio database if needed."""
    try:
        # Import the seeding function
        import sys
        from pathlib import Path

        # Try different paths for the scripts directory
        possible_paths = [
            Path(__file__).parent.parent.parent / "scripts",  # Local development
            Path.cwd() / "scripts",  # Docker environment
            Path("/app/scripts") if Path("/app/scripts").exists() else None,  # Docker absolute path
        ]
        
        scripts_path = None
        for path in possible_paths:
            if path and path.exists():
                scripts_path = path
                break
        
        if not scripts_path:
            raise FileNotFoundError("Could not find scripts directory")
        
        sys.path.insert(0, str(scripts_path))
        
        from seed_portfolio_db import seed_portfolio_database

        # Run seeding (it will check if needed internally)
        seed_portfolio_database()
        
    except Exception as e:
        # Log error but don't fail the service startup
        print(f"Warning: Portfolio database auto-seeding failed: {e}")
        # In production, you might want to use proper logging instead of print


__all__ = ["initialize_database", "initialize_store", "initialize_portfolio_database"]
