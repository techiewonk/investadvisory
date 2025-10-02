# Portfolio APIs Documentation

## Overview

The Investment Advisory AI Platform now includes comprehensive portfolio management APIs that allow you to retrieve client portfolio data, transaction history, and perform portfolio analysis.

## Database Schema

The portfolio system uses the following database tables:

- **`user`**: Client information and risk profiles
- **`security`**: Financial instruments (stocks, bonds, ETFs, etc.)
- **`portfolio`**: Client portfolios
- **`history`**: Transaction records (BUY/SELL)

## API Endpoints

### 1. Get All Clients

Retrieve a list of all clients in the system with summary information.

#### GET `/portfolio/clients`

**Response:**

```json
{
  "clients": [
    {
      "client_id": "CLT-001",
      "name": "John Doe",
      "email": "john@example.com",
      "risk_profile": "Balanced",
      "portfolio_count": 1,
      "total_portfolio_value": 15000.0,
      "created_at": "2024-01-01T00:00:00Z"
    },
    {
      "client_id": "CLT-002",
      "name": "Jane Smith",
      "email": "jane@example.com",
      "risk_profile": "Aggressive",
      "portfolio_count": 2,
      "total_portfolio_value": 25000.0,
      "created_at": "2024-01-02T00:00:00Z"
    }
  ],
  "total_clients": 2
}
```

### 2. Get User Portfolios

Retrieve all portfolios for a specific client.

#### GET `/portfolio/users/{client_id}/portfolios`

**Parameters:**

- `client_id` (path): Business client reference ID (e.g., "CLT-001")

**Response:**

```json
{
  "user": {
    "id": 1,
    "client_id": "CLT-001",
    "name": "John Doe",
    "email": "john@example.com",
    "risk_profile": "Balanced",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "portfolios": [
    {
      "portfolio": {
        "id": 1,
        "user_id": 1,
        "name": "Default Portfolio for CLT-001",
        "created_at": "2024-01-01T00:00:00Z"
      },
      "user": {
        /* user object */
      },
      "holdings": [
        {
          "security": {
            "id": 1,
            "symbol": "AAPL",
            "security_name": "Apple Inc.",
            "asset_class": "Stock",
            "sector": "Technology"
          },
          "total_quantity": 100,
          "average_price": 150.0,
          "total_value": 15000.0,
          "unrealized_pnl": 0.0
        }
      ],
      "total_value": 15000.0,
      "total_cost": 15000.0,
      "unrealized_pnl": 0.0
    }
  ]
}
```

#### POST `/portfolio/user-portfolios`

**Request Body:**

```json
{
  "client_id": "CLT-001"
}
```

### 2. Get User Transactions

Retrieve transaction history for a specific client.

#### GET `/portfolio/users/{client_id}/transactions`

**Parameters:**

- `client_id` (path): Business client reference ID
- `limit` (query, optional): Maximum transactions to return (default: 100)
- `offset` (query, optional): Number of transactions to skip (default: 0)

**Response:**

```json
{
  "user": {
    /* user object */
  },
  "transactions": [
    {
      "transaction": {
        "id": 1,
        "portfolio_id": 1,
        "security_id": 1,
        "transaction_type": "BUY",
        "quantity": 100,
        "price": 150.0,
        "transaction_date": "2024-01-01",
        "notes": null
      },
      "security": {
        "id": 1,
        "symbol": "AAPL",
        "security_name": "Apple Inc.",
        "asset_class": "Stock",
        "sector": "Technology"
      }
    }
  ],
  "total_transactions": 1
}
```

#### POST `/portfolio/user-transactions`

**Request Body:**

```json
{
  "client_id": "CLT-001",
  "limit": 50,
  "offset": 0
}
```

## Agent Tools

The platform includes specialized tools that AI agents can use to access portfolio data:

### 1. `get_all_clients()`

Retrieves a list of all clients in the system with summary information including portfolio counts and total values.

### 2. `get_client_portfolios(client_id: str)`

Retrieves portfolio information formatted for agent consumption.

### 3. `get_client_transactions(client_id: str, limit: int = 10)`

Gets recent transaction history for analysis.

### 4. `analyze_client_portfolio_performance(client_id: str)`

Performs comprehensive portfolio analysis including:

- Performance metrics
- Sector diversification
- Top holdings analysis
- Risk assessment

## Database Setup

1. **Create the database schema:**

   ```bash
   # Use the provided db_schema.sql file
   psql -d your_database -f db_schema.sql
   ```

2. **Seed with sample data:**
   ```bash
   # Use the seed script with your CSV data
   python scripts/seed_portfolio_db.py
   ```

## Configuration

The portfolio service requires PostgreSQL configuration in your environment:

```env
DATABASE_TYPE=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=investadvisory
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

## Testing

Use the provided test script to verify the APIs:

```bash
python test_portfolio_api.py
```

## Error Handling

The APIs return appropriate HTTP status codes:

- **200**: Success
- **404**: Client not found
- **400**: Invalid request parameters
- **500**: Internal server error

All errors include descriptive messages in the response body.

## Integration with AI Agents

The portfolio tools can be easily integrated into any LangGraph agent:

```python
from agents.portfolio_tools import PORTFOLIO_TOOLS

# Add to your agent's tools
agent = create_react_agent(
    model=model,
    tools=[...other_tools, *PORTFOLIO_TOOLS],
    prompt="You are an investment advisor with access to client portfolio data..."
)
```

## Security Considerations

- All endpoints require authentication (AUTH_SECRET header)
- Client data is accessed by client_id only
- No sensitive financial data is logged
- Database connections use connection pooling for security and performance
