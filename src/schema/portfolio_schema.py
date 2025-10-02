"""Portfolio-related schemas for investment advisory platform."""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

# Constants
CLIENT_ID_DESCRIPTION = "Business client reference ID"


class UserProfile(BaseModel):
    """User profile information."""
    
    id: int = Field(description="Internal user ID")
    client_id: str = Field(description=CLIENT_ID_DESCRIPTION, examples=["CLT-001"])
    name: str | None = Field(description="User's full name", default=None)
    email: str | None = Field(description="User's email address", default=None)
    risk_profile: str | None = Field(
        description="Investment risk profile", 
        examples=["Conservative", "Balanced", "Aggressive"],
        default=None
    )
    created_at: datetime = Field(description="Account creation timestamp")


class Security(BaseModel):
    """Security/instrument information."""
    
    id: int = Field(description="Internal security ID")
    symbol: str = Field(description="Trading symbol", examples=["AAPL", "MSFT"])
    security_name: str = Field(description="Full security name", examples=["Apple Inc."])
    asset_class: str | None = Field(
        description="Asset class", 
        examples=["Stock", "Bond", "ETF"],
        default=None
    )
    sector: str | None = Field(description="Industry sector", default=None)
    exchange: str | None = Field(
        description="Trading exchange", 
        examples=["NYSE", "NASDAQ"],
        default=None
    )
    currency: str | None = Field(description="Trading currency", examples=["USD", "EUR"], default=None)
    isin: str | None = Field(description="International Securities Identification Number", default=None)
    description: str | None = Field(description="Security description", default=None)


class Portfolio(BaseModel):
    """Portfolio information."""
    
    id: int = Field(description="Internal portfolio ID")
    user_id: int = Field(description="Owner user ID")
    name: str = Field(description="Portfolio name", examples=["Retirement", "Growth Portfolio"])
    created_at: datetime = Field(description="Portfolio creation timestamp")


class Transaction(BaseModel):
    """Transaction/history record."""
    
    id: int = Field(description="Internal transaction ID")
    portfolio_id: int = Field(description="Portfolio ID")
    security_id: int = Field(description="Security ID")
    transaction_type: Literal["BUY", "SELL"] = Field(description="Transaction type")
    quantity: Decimal = Field(description="Number of shares/units")
    price: Decimal = Field(description="Price per share/unit")
    transaction_date: date = Field(description="Transaction date")
    notes: str | None = Field(description="Additional notes", default=None)


class PortfolioHolding(BaseModel):
    """Current portfolio holding with security details."""
    
    security: Security = Field(description="Security information")
    total_quantity: Decimal = Field(description="Total shares owned")
    average_price: Decimal = Field(description="Average purchase price")
    total_value: Decimal = Field(description="Current total value")
    unrealized_pnl: Decimal | None = Field(description="Unrealized profit/loss", default=None)


class PortfolioSummary(BaseModel):
    """Portfolio summary with holdings."""
    
    portfolio: Portfolio = Field(description="Portfolio information")
    user: UserProfile = Field(description="Portfolio owner information")
    holdings: list[PortfolioHolding] = Field(description="Current holdings")
    total_value: Decimal = Field(description="Total portfolio value")
    total_cost: Decimal = Field(description="Total cost basis")
    unrealized_pnl: Decimal = Field(description="Total unrealized profit/loss")


class UserPortfoliosResponse(BaseModel):
    """Response containing all portfolios for a user."""
    
    user: UserProfile = Field(description="User information")
    portfolios: list[PortfolioSummary] = Field(description="User's portfolios")


class TransactionHistory(BaseModel):
    """Transaction history with security details."""
    
    transaction: Transaction = Field(description="Transaction details")
    security: Security = Field(description="Security information")


class UserTransactionsResponse(BaseModel):
    """Response containing transaction history for a user."""
    
    user: UserProfile = Field(description="User information")
    transactions: list[TransactionHistory] = Field(description="Transaction history")
    total_transactions: int = Field(description="Total number of transactions")


# Request models
class GetUserPortfoliosRequest(BaseModel):
    """Request to get user portfolios by client ID."""
    
    client_id: str = Field(description=CLIENT_ID_DESCRIPTION, examples=["CLT-001"])


class GetUserTransactionsRequest(BaseModel):
    """Request to get user transactions by client ID."""
    
    client_id: str = Field(description=CLIENT_ID_DESCRIPTION, examples=["CLT-001"])
    limit: int = Field(description="Maximum number of transactions to return", default=100, ge=1, le=1000)
    offset: int = Field(description="Number of transactions to skip", default=0, ge=0)


class GetPortfolioDetailsRequest(BaseModel):
    """Request to get detailed portfolio information."""
    
    portfolio_id: int = Field(description="Portfolio ID")


class ClientSummary(BaseModel):
    """Summary information for a client."""
    
    client_id: str = Field(description=CLIENT_ID_DESCRIPTION, examples=["CLT-001"])
    name: str | None = Field(description="Client name", default=None)
    email: str | None = Field(description="Client email", default=None)
    risk_profile: str | None = Field(description="Risk profile", default=None)
    portfolio_count: int = Field(description="Number of portfolios")
    total_portfolio_value: Decimal = Field(description="Total value across all portfolios")
    created_at: datetime = Field(description="Account creation date")


class AllClientsResponse(BaseModel):
    """Response containing all clients in the system."""
    
    clients: list[ClientSummary] = Field(description="List of all clients")
    total_clients: int = Field(description="Total number of clients")
