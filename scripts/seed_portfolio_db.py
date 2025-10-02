
import pandas as pd
from sqlalchemy import create_engine, text

# CONFIGURE THIS FOR YOUR ENVIRONMENT
DB_URL = "sqlite:///core_database.db"  # Or your Postgres URL
CSV_PATH = "portfolios.csv"

# Read CSV
df = pd.read_csv(CSV_PATH)

# Clean up column names and values
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df['purchase_price'] = df['purchase_price'].replace(r'[^0-9.]', '', regex=True).astype(float)


# Read schema from db_schema.sql
with open("db_schema.sql", "r", encoding="utf-8") as f:
    schema_sql = f.read()

engine = create_engine(DB_URL)


with engine.begin() as conn:
    # Create schema if not exists
    for stmt in schema_sql.strip().split(';'):
        if stmt.strip():
            conn.execute(text(stmt))

    # Delete from tables in correct dependency order for new schema (after all tables exist)
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

    # Create one portfolio per user (e.g., 'Default Portfolio for CLT-001')
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

print("Database seeded from portfolios.csv!")
