"""Portfolio database service for investment advisory platform."""

import logging
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

from core.settings import DatabaseType, settings
from memory import initialize_portfolio_database
from schema.portfolio_schema import (
    AllClientsResponse,
    ClientSummary,
    Portfolio,
    PortfolioHolding,
    PortfolioSummary,
    Security,
    Transaction,
    TransactionHistory,
    UserPortfoliosResponse,
    UserProfile,
    UserTransactionsResponse,
)

logger = logging.getLogger(__name__)


class PortfolioService:
    """Generic portfolio service that works with any database connection."""
    
    def __init__(self, db_connection: Any):
        self.db_connection = db_connection
    
    def _format_query(self, query: str, param_count: int) -> str:
        """Format query with appropriate parameter placeholders."""
        if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
            # Replace ? with $1, $2, etc. for PostgreSQL
            formatted_query = query
            for i in range(1, param_count + 1):
                formatted_query = formatted_query.replace("?", f"${i}", 1)
            return formatted_query
        return query
    
    async def _execute_query(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as list of dictionaries."""
        formatted_query = self._format_query(query, len(params))
        
        if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
            # PostgreSQL connection pool
            async with self.db_connection.connection() as conn:
                rows = await conn.fetch(formatted_query, *params)
                return [dict(row) for row in rows]
        else:
            # SQLite connection
            cursor = await self.db_connection.execute(formatted_query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def _execute_query_one(self, query: str, params: tuple = ()) -> dict | None:
        """Execute a query and return single result as dictionary."""
        formatted_query = self._format_query(query, len(params))
        
        if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
            # PostgreSQL connection pool
            async with self.db_connection.connection() as conn:
                row = await conn.fetchrow(formatted_query, *params)
                return dict(row) if row else None
        else:
            # SQLite connection
            cursor = await self.db_connection.execute(formatted_query, params)
            row = await cursor.fetchone()
            return dict(row) if row else None
    
    async def get_user_by_client_id(self, client_id: str) -> UserProfile | None:
        """Get user by client ID."""
        query = """
            SELECT id, client_id, name, email, risk_profile, created_at
            FROM user 
            WHERE client_id = ?
        """
        
        row = await self._execute_query_one(query, (client_id,))
        if row:
            return UserProfile(**row)
        return None
    
    async def get_user_portfolios(self, client_id: str) -> UserPortfoliosResponse | None:
        """Get all portfolios for a user by client ID."""
        user = await self.get_user_by_client_id(client_id)
        if not user:
            return None
        
        # Get portfolios
        portfolio_query = """
            SELECT id, user_id, name, created_at
            FROM portfolio 
            WHERE user_id = ?
            ORDER BY created_at DESC
        """
        
        portfolio_rows = await self._execute_query(portfolio_query, (user.id,))
        portfolios = []
        
        for portfolio_row in portfolio_rows:
            portfolio = Portfolio(**portfolio_row)
            
            # Get holdings for this portfolio
            holdings = await self._get_portfolio_holdings(portfolio.id)
            
            # Calculate totals
            total_value = Decimal(str(sum(h.total_value for h in holdings)))
            total_cost = Decimal(str(sum(h.average_price * h.total_quantity for h in holdings)))
            unrealized_pnl = total_value - total_cost
            
            portfolio_summary = PortfolioSummary(
                portfolio=portfolio,
                user=user,
                holdings=holdings,
                total_value=total_value,
                total_cost=total_cost,
                unrealized_pnl=unrealized_pnl
            )
            portfolios.append(portfolio_summary)
        
        return UserPortfoliosResponse(user=user, portfolios=portfolios)
    
    async def _get_portfolio_holdings(self, portfolio_id: int) -> list[PortfolioHolding]:
        """Get current holdings for a portfolio."""
        holdings_query = """
            SELECT 
                s.id, s.symbol, s.security_name, s.asset_class, s.sector, 
                s.exchange, s.currency, s.isin, s.description,
                SUM(CASE WHEN h.transaction_type = 'BUY' THEN h.quantity 
                         WHEN h.transaction_type = 'SELL' THEN -h.quantity 
                         ELSE 0 END) as total_quantity,
                AVG(CASE WHEN h.transaction_type = 'BUY' THEN h.price ELSE NULL END) as avg_price
            FROM history h
            JOIN security s ON h.security_id = s.id
            WHERE h.portfolio_id = ?
            GROUP BY s.id, s.symbol, s.security_name, s.asset_class, s.sector, 
                     s.exchange, s.currency, s.isin, s.description
            HAVING SUM(CASE WHEN h.transaction_type = 'BUY' THEN h.quantity 
                           WHEN h.transaction_type = 'SELL' THEN -h.quantity 
                           ELSE 0 END) > 0
        """
        
        rows = await self._execute_query(holdings_query, (portfolio_id,))
        holdings = []
        
        for row in rows:
            security = Security(
                id=row['id'],
                symbol=row['symbol'],
                security_name=row['security_name'],
                asset_class=row['asset_class'],
                sector=row['sector'],
                exchange=row['exchange'],
                currency=row['currency'],
                isin=row['isin'],
                description=row['description']
            )
            
            total_quantity = Decimal(str(row['total_quantity']))
            average_price = Decimal(str(row['avg_price'] or 0))
            
            # For now, use average price as current price (in real system, get from market data)
            current_price = average_price
            total_value = total_quantity * current_price
            
            holding = PortfolioHolding(
                security=security,
                total_quantity=total_quantity,
                average_price=average_price,
                total_value=total_value,
                unrealized_pnl=total_value - (average_price * total_quantity)
            )
            holdings.append(holding)
        
        return holdings
    
    async def get_user_transactions(
        self, 
        client_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> UserTransactionsResponse | None:
        """Get transaction history for a user by client ID."""
        user = await self.get_user_by_client_id(client_id)
        if not user:
            return None
        
        # Get transactions with security details
        transactions_query = """
            SELECT 
                h.id, h.portfolio_id, h.security_id, h.transaction_type,
                h.quantity, h.price, h.transaction_date, h.notes,
                s.id as sec_id, s.symbol, s.security_name, s.asset_class, 
                s.sector, s.exchange, s.currency, s.isin, s.description
            FROM history h
            JOIN portfolio p ON h.portfolio_id = p.id
            JOIN security s ON h.security_id = s.id
            WHERE p.user_id = ?
            ORDER BY h.transaction_date DESC, h.id DESC
            LIMIT ? OFFSET ?
        """
        
        # Get total count
        count_query = """
            SELECT COUNT(*) as total
            FROM history h
            JOIN portfolio p ON h.portfolio_id = p.id
            WHERE p.user_id = ?
        """
        
        transaction_rows = await self._execute_query(transactions_query, (user.id, limit, offset))
        count_result = await self._execute_query_one(count_query, (user.id,))
        total_transactions = count_result['total'] if count_result else 0
        
        transactions = []
        for row in transaction_rows:
            security = Security(
                id=row['sec_id'],
                symbol=row['symbol'],
                security_name=row['security_name'],
                asset_class=row['asset_class'],
                sector=row['sector'],
                exchange=row['exchange'],
                currency=row['currency'],
                isin=row['isin'],
                description=row['description']
            )
            
            transaction = Transaction(
                id=row['id'],
                portfolio_id=row['portfolio_id'],
                security_id=row['security_id'],
                transaction_type=row['transaction_type'],
                quantity=Decimal(str(row['quantity'])),
                price=Decimal(str(row['price'])),
                transaction_date=row['transaction_date'],
                notes=row['notes']
            )
            
            transaction_history = TransactionHistory(
                transaction=transaction,
                security=security
            )
            transactions.append(transaction_history)
        
        return UserTransactionsResponse(
            user=user,
            transactions=transactions,
            total_transactions=total_transactions
        )
    
    async def get_all_clients(self) -> AllClientsResponse:
        """Get all clients with summary information."""
        clients_query = """
            SELECT 
                u.id, u.client_id, u.name, u.email, u.risk_profile, u.created_at,
                COUNT(p.id) as portfolio_count,
                COALESCE(SUM(portfolio_values.total_value), 0) as total_portfolio_value
            FROM user u
            LEFT JOIN portfolio p ON u.id = p.user_id
            LEFT JOIN (
                SELECT 
                    p.id as portfolio_id,
                    SUM(
                        CASE WHEN h.transaction_type = 'BUY' THEN h.quantity * h.price
                             WHEN h.transaction_type = 'SELL' THEN -h.quantity * h.price
                             ELSE 0 END
                    ) as total_value
                FROM portfolio p
                LEFT JOIN history h ON p.id = h.portfolio_id
                GROUP BY p.id
            ) portfolio_values ON p.id = portfolio_values.portfolio_id
            GROUP BY u.id, u.client_id, u.name, u.email, u.risk_profile, u.created_at
            ORDER BY u.client_id
        """
        
        rows = await self._execute_query(clients_query)
        clients = []
        
        for row in rows:
            client_summary = ClientSummary(
                client_id=row['client_id'],
                name=row['name'],
                email=row['email'],
                risk_profile=row['risk_profile'],
                portfolio_count=row['portfolio_count'],
                total_portfolio_value=Decimal(str(row['total_portfolio_value'] or 0)),
                created_at=row['created_at']
            )
            clients.append(client_summary)
        
        return AllClientsResponse(
            clients=clients,
            total_clients=len(clients)
        )


@asynccontextmanager
async def get_portfolio_service():
    """Get portfolio service with database connection based on settings."""
    async with initialize_portfolio_database() as db_connection:
        service = PortfolioService(db_connection)
        yield service
