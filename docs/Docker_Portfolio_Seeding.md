# Docker Portfolio Seeding Configuration

This document explains how to configure automatic portfolio seeding in Docker environments for the Investment Advisory AI Platform.

## Quick Start

### 1. Basic Docker Setup with Auto-Seeding

```bash
# Create your .env file
cp .env.example .env

# Edit .env to include:
DATABASE_TYPE=postgres
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=investadvisory
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
SEED_PORTFOLIO_DB=true

# Start with portfolio seeding enabled
docker compose up --build
```

### 2. Environment Variable Control

You can control seeding behavior using environment variables:

```bash
# Enable seeding (default in Docker)
export SEED_PORTFOLIO_DB=true
docker compose up --build

# Disable seeding for production
export SEED_PORTFOLIO_DB=false
docker compose up --build

# One-time override
SEED_PORTFOLIO_DB=true docker compose up --build
```

## Configuration Details

### Docker Compose Configuration

The `compose.yaml` includes automatic portfolio seeding configuration:

```yaml
agent_service:
  environment:
    # Enable portfolio seeding in Docker environment
    - SEED_PORTFOLIO_DB=${SEED_PORTFOLIO_DB:-true}
  # ... other config
```

**Default Behavior:**

- `SEED_PORTFOLIO_DB=true` by default in Docker
- Automatically loads sample portfolio data on first startup
- Safe to run multiple times (checks for existing data)

### Database Configuration

#### PostgreSQL (Recommended for Docker)

```env
DATABASE_TYPE=postgres
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=investadvisory
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
SEED_PORTFOLIO_DB=true
```

#### SQLite (Alternative)

```env
DATABASE_TYPE=sqlite
SQLITE_DB_PATH=checkpoints.db
# SEED_PORTFOLIO_DB not needed (always enabled for SQLite)
```

## Seeding Behavior in Docker

### ✅ **With SEED_PORTFOLIO_DB=true (Default)**

**Container Startup Output:**

```
Portfolio Database Seeding
   Environment: postgres
   CSV: portfolios.csv
   DB: postgresql://...
   Processing 69 portfolio entries...
   Successfully seeded 69 portfolio entries!
      - 8 clients
      - 54 securities
      - 69 transactions
```

**Benefits:**

- Immediate access to realistic portfolio data
- Perfect for development and testing
- Demonstrates full platform capabilities

### ⏭️ **With SEED_PORTFOLIO_DB=false**

**Container Startup Output:**

```
Portfolio Database Seeding
   Environment: postgres
   Skipping seeding (not enabled for postgres)
```

**Use Cases:**

- Production deployments with real client data
- Custom data loading scenarios
- Clean database for testing

## Sample Data Included

When seeding is enabled, your Docker environment automatically includes:

### 📊 **Client Portfolios**

- **CLT-001**: Balanced ETF portfolio (VTI, BND, VXUS, VYM, VTEB)
- **CLT-002**: Tech-focused growth (QQQ, NVDA, MSFT, ARKK, TSLA)
- **CLT-003**: Traditional index funds (SPY, AGG, EFA, VWO)
- **CLT-004**: Small-cap growth (SCHG, VB, VXUS)
- **CLT-005**: Individual tech stocks (AAPL, MSFT, GOOGL, NVDA, META)
- **CLT-007**: Value/dividend focus (BRK.B, BAC, XOM, WMT)
- **CLT-009**: High-growth tech (SHOP, ROKU, PLTR, CRWD)
- **CLT-010**: ESG/sustainable investing (VSGX, ICLN, NEE, ENPH)

### 🏢 **Asset Classes**

- **Stocks**: Individual companies (AAPL, MSFT, TSLA, etc.)
- **ETFs**: Broad market, sector, and thematic funds
- **Bonds**: Government and corporate bond funds
- **Cash**: Money market and high-yield savings

### 📈 **Realistic Data**

- **69 transactions** across all portfolios
- **Multiple sectors**: Technology, Healthcare, Energy, Financials
- **Diverse strategies**: Growth, value, ESG, dividend focus
- **Various time periods**: Recent and historical purchases

## Docker Development Workflow

### 1. **Development with Live Reloading**

```bash
# Start with auto-seeding and file watching
docker compose up --build --watch
```

### 2. **Clean Database Reset**

```bash
# Stop services and remove volumes
docker compose down -v

# Restart with fresh seeding
docker compose up --build
```

### 3. **Production-like Testing**

```bash
# Disable seeding for production simulation
SEED_PORTFOLIO_DB=false docker compose up --build

# Manually seed with custom data
docker compose exec agent_service python scripts/seed_portfolio_db.py
```

## Troubleshooting

### Issue: Seeding Fails in Docker

**Check:**

1. Portfolio files are copied to container:

   ```bash
   docker compose exec agent_service ls -la portfolios.csv db_schema.sql scripts/
   ```

2. Database connection is working:

   ```bash
   docker compose exec agent_service python -c "from core.settings import settings; print(settings.DATABASE_TYPE)"
   ```

3. Environment variables are set:
   ```bash
   docker compose exec agent_service env | grep SEED_PORTFOLIO_DB
   ```

### Issue: Data Not Appearing

**Solutions:**

1. Check if seeding was skipped (data already exists)
2. Verify database connection settings
3. Check container logs for seeding output:
   ```bash
   docker compose logs agent_service | grep "Portfolio Database"
   ```

## File Structure in Container

```
/app/
├── agents/              # AI agent modules
├── core/               # Core settings and LLM
├── memory/             # Database connections
├── schema/             # Data models
├── service/            # API service
├── scripts/            # Seeding scripts
│   └── seed_portfolio_db.py
├── portfolios.csv      # Sample portfolio data
├── db_schema.sql       # Database schema
└── run_service.py      # Service entry point
```

## Best Practices

### 🔧 **Development**

- Use `SEED_PORTFOLIO_DB=true` (default)
- Leverage Docker Compose watch for live reloading
- Reset volumes periodically for clean state

### 🚀 **Production**

- Set `SEED_PORTFOLIO_DB=false` for safety
- Use real client data instead of samples
- Monitor seeding logs during deployment

### 🧪 **Testing**

- Use seeded data for consistent test scenarios
- Reset database between test runs
- Verify API responses with known sample data

Your Docker environment now supports intelligent portfolio seeding that adapts to your deployment needs! 🐳
