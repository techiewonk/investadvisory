
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.settings import DatabaseType, settings


def get_database_url() -> str:
    """Get database URL based on environment settings."""
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        # PostgreSQL URL for production
        return (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
    else:
        # SQLite URL for development - use portfolio-specific database
        portfolio_db_path = settings.SQLITE_DB_PATH.replace('.db', '_portfolio.db')
        return f"sqlite:///{portfolio_db_path}"


def get_csv_path() -> str:
    """Get CSV file path."""
    script_dir = Path(__file__).parent.parent
    csv_path = script_dir / "portfolios.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Portfolio CSV file not found: {csv_path}")
    return str(csv_path)


def should_seed_database() -> bool:
    """Check if database should be seeded based on environment."""
    # Always seed in development (SQLite)
    if settings.DATABASE_TYPE == DatabaseType.SQLITE:
        return True
    
    # In production (PostgreSQL), only seed if explicitly requested
    return os.getenv("SEED_PORTFOLIO_DB", "false").lower() in ("true", "1", "yes")


def check_if_already_seeded(engine) -> bool:
    """Check if database already has portfolio data."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM user"))
            count = result.fetchone()[0]
            return count > 0
    except Exception:
        # If table doesn't exist or any error, assume not seeded
        return False


def seed_portfolio_database():
    """Main function to seed the portfolio database."""
    print(f"Portfolio Database Seeding")
    print(f"   Environment: {settings.DATABASE_TYPE.value}")
    
    # Check if seeding should happen
    if not should_seed_database():
        print(f"   Skipping seeding (not enabled for {settings.DATABASE_TYPE.value})")
        return
    
    try:
        # Get database connection
        db_url = get_database_url()
        csv_path = get_csv_path()
        
        print(f"   CSV: {Path(csv_path).name}")
        print(f"   DB: {db_url.split('://')[0]}://...")
        
        # Create engine
        engine = create_engine(db_url)
        
        # Check if already seeded
        if check_if_already_seeded(engine):
            print(f"   Database already contains data, skipping seeding")
            return
        
        # Read and process CSV
        df = pd.read_csv(csv_path)
        print(f"   Processing {len(df)} portfolio entries...")
        
        # Clean up column names and values
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        df['purchase_price'] = df['purchase_price'].replace(r'[^0-9.]', '', regex=True).astype(float)
        
        # Read schema
        schema_path = Path(__file__).parent.parent / "db_schema.sql"
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        
        # Execute seeding
        with engine.begin() as conn:
            # Create schema if not exists
            for stmt in schema_sql.strip().split(';'):
                if stmt.strip():
                    conn.execute(text(stmt))
            
            # Clear existing data (in case of re-seeding)
            conn.execute(text("DELETE FROM history"))
            conn.execute(text("DELETE FROM portfolio"))
            conn.execute(text("DELETE FROM user"))
            conn.execute(text("DELETE FROM security"))
            
            # Insert users (deduplicate by client_id)
            users = df[['client_id']].drop_duplicates().copy()
            users['name'] = None
            users['email'] = None
            users['risk_profile'] = None
            users.to_sql('user', conn, if_exists='append', index=False)
            user_map = pd.read_sql(
                'SELECT id, client_id FROM user', conn
            ).set_index('client_id')['id'].to_dict()
            
            # Insert securities (deduplicate by symbol+security_name)
            securities = df[['symbol', 'security_name', 'asset_class', 'sector']]
            securities = securities.drop_duplicates(subset=['symbol', 'security_name']).copy()
            securities['exchange'] = None
            securities['currency'] = None
            securities['isin'] = None
            securities['description'] = None
            securities.to_sql('security', conn, if_exists='append', index=False)
            security_map = pd.read_sql(
                'SELECT id, symbol, security_name FROM security', conn
            ).set_index(['symbol', 'security_name'])['id'].to_dict()
            
            # Create one portfolio per user
            portfolios = users[['client_id']].copy()
            portfolios['user_id'] = portfolios['client_id'].map(user_map)
            portfolios['name'] = portfolios['client_id'].apply(lambda cid: f"Default Portfolio for {cid}")
            portfolios = portfolios[['user_id', 'name']]
            portfolios.to_sql('portfolio', conn, if_exists='append', index=False)
            portfolio_map = pd.read_sql(
                'SELECT id, user_id, name FROM portfolio', conn
            ).set_index(['user_id', 'name'])['id'].to_dict()
            
            # Insert all transactions as history (assume all are BUY for now)
            history_rows = []
            for _, row in df.iterrows():
                user_id = user_map[row['client_id']]
                PNAME = f"Default Portfolio for {row['client_id']}"
                portfolio_id = portfolio_map[(user_id, PNAME)]
                security_id = security_map[(row['symbol'], row['security_name'])]
                history_rows.append({
                    'portfolio_id': portfolio_id,
                    'security_id': security_id,
                    'transaction_type': 'BUY',
                    'quantity': row['quantity'],
                    'price': row['purchase_price'],
                    'transaction_date': row['purchase_date'],
                    'notes': None
                })
            if history_rows:
                pd.DataFrame(history_rows).to_sql('history', conn, if_exists='append', index=False)
        
        print(f"   Successfully seeded {len(df)} portfolio entries!")
        print(f"      - {len(users)} clients")
        print(f"      - {len(securities)} securities") 
        print(f"      - {len(history_rows)} transactions")
        
    except Exception as e:
        print(f"   Error seeding database: {e}")
        raise


if __name__ == "__main__":
    seed_portfolio_database()
