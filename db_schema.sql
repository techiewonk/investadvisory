-- Many-to-many relationship between user and portfolio


CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT UNIQUE NOT NULL,         -- Business/client reference, must be unique
    name TEXT,
    email TEXT,
    risk_profile TEXT,             -- e.g., "Conservative", "Balanced", "Aggressive"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS security (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    security_name TEXT NOT NULL,
    asset_class TEXT,          -- e.g., "Stock", "Bond", "ETF", etc.
    sector TEXT,
    exchange TEXT,             -- e.g., "NYSE", "NASDAQ"
    currency TEXT,             -- e.g., "USD", "EUR"
    isin TEXT,                 -- International Securities Identification Number
    description TEXT,
    UNIQUE(symbol, security_name)
);

CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,                 -- e.g., "Retirement", "Growth Portfolio"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES user(id)
);

-- History table: all transactions (BUY/SELL), current and past
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    security_id INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,     -- "BUY", "SELL", etc.
    quantity NUMERIC,
    price NUMERIC,
    transaction_date DATE,
    notes TEXT,
    FOREIGN KEY(portfolio_id) REFERENCES portfolio(id),
    FOREIGN KEY(security_id) REFERENCES security(id)
);
