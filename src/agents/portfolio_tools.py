"""Portfolio tools for investment advisory agents."""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from service.portfolio_service import get_portfolio_service

logger = logging.getLogger(__name__)


def get_selected_client_id(config: RunnableConfig | None = None) -> str | None:
    """Extract selected client ID from agent config."""
    if not config or not config.get("configurable"):
        return None
    
    selected_client = config["configurable"].get("selected_client")
    if selected_client and isinstance(selected_client, dict):
        return selected_client.get("client_id")
    
    return None


@tool
async def get_client_portfolios(client_id: str) -> dict[str, Any]:
    """
    Get portfolio information for a client by their client ID.
    
    Args:
        client_id: The client's unique identifier (e.g., 'CLT-001')
        
    Returns:
        Dictionary containing client portfolios, holdings, and performance data
    """
    try:
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Convert to dictionary for agent consumption
            return {
                "client": {
                    "id": result.user.client_id,
                    "name": result.user.name,
                    "risk_profile": result.user.risk_profile,
                },
                "portfolios": [
                    {
                        "name": p.portfolio.name,
                        "total_value": float(p.total_value),
                        "total_cost": float(p.total_cost),
                        "unrealized_pnl": float(p.unrealized_pnl),
                        "holdings_count": len(p.holdings),
                        "top_holdings": [
                            {
                                "symbol": h.security.symbol,
                                "name": h.security.security_name,
                                "quantity": float(h.total_quantity),
                                "value": float(h.total_value),
                                "sector": h.security.sector,
                            }
                            for h in sorted(p.holdings, key=lambda x: x.total_value, reverse=True)[:5]
                        ]
                    }
                    for p in result.portfolios
                ]
            }
    except Exception as e:
        logger.error(f"Error getting client portfolios: {e}")
        return {"error": f"Failed to retrieve portfolio data: {str(e)}"}


@tool
async def get_client_transactions(client_id: str, limit: int = 10) -> dict[str, Any]:
    """
    Get recent transaction history for a client.
    
    Args:
        client_id: The client's unique identifier (e.g., 'CLT-001')
        limit: Maximum number of transactions to return (default: 10)
        
    Returns:
        Dictionary containing recent transactions
    """
    try:
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_transactions(client_id, limit=limit, offset=0)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            return {
                "client_id": result.user.client_id,
                "total_transactions": result.total_transactions,
                "recent_transactions": [
                    {
                        "date": t.transaction.transaction_date.isoformat(),
                        "type": t.transaction.transaction_type,
                        "symbol": t.security.symbol,
                        "security_name": t.security.security_name,
                        "quantity": float(t.transaction.quantity),
                        "price": float(t.transaction.price),
                        "total_amount": float(t.transaction.quantity * t.transaction.price),
                        "sector": t.security.sector,
                    }
                    for t in result.transactions
                ]
            }
    except Exception as e:
        logger.error(f"Error getting client transactions: {e}")
        return {"error": f"Failed to retrieve transaction data: {str(e)}"}


@tool
async def get_all_clients() -> dict[str, Any]:
    """
    Get a list of all clients in the system with summary information.
    
    Returns:
        Dictionary containing all clients with their basic info and portfolio summaries
    """
    try:
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_all_clients()
            
            return {
                "total_clients": result.total_clients,
                "clients": [
                    {
                        "client_id": client.client_id,
                        "name": client.name,
                        "email": client.email,
                        "risk_profile": client.risk_profile,
                        "portfolio_count": client.portfolio_count,
                        "total_portfolio_value": float(client.total_portfolio_value),
                        "created_at": client.created_at.isoformat(),
                    }
                    for client in result.clients
                ]
            }
    except Exception as e:
        logger.error(f"Error getting all clients: {e}")
        return {"error": f"Failed to retrieve clients: {str(e)}"}


@tool
async def analyze_client_portfolio_performance(client_id: str) -> dict[str, Any]:
    """
    Analyze portfolio performance and provide investment insights for a client.
    
    Args:
        client_id: The client's unique identifier (e.g., 'CLT-001')
        
    Returns:
        Dictionary containing portfolio analysis and recommendations
    """
    try:
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            analysis = {
                "client_id": client_id,
                "risk_profile": result.user.risk_profile,
                "portfolio_analysis": []
            }
            
            for portfolio in result.portfolios:
                # Calculate performance metrics
                total_return_pct = (float(portfolio.unrealized_pnl) / float(portfolio.total_cost) * 100) if portfolio.total_cost > 0 else 0
                
                # Sector diversification
                sector_allocation = {}
                for holding in portfolio.holdings:
                    sector = holding.security.sector or "Unknown"
                    if sector not in sector_allocation:
                        sector_allocation[sector] = 0
                    sector_allocation[sector] += float(holding.total_value)
                
                # Calculate percentages
                total_value = float(portfolio.total_value)
                sector_percentages = {
                    sector: (value / total_value * 100) if total_value > 0 else 0
                    for sector, value in sector_allocation.items()
                }
                
                portfolio_analysis = {
                    "name": portfolio.portfolio.name,
                    "total_value": total_value,
                    "total_return_percentage": round(total_return_pct, 2),
                    "holdings_count": len(portfolio.holdings),
                    "sector_allocation": sector_percentages,
                    "top_performers": [
                        {
                            "symbol": h.security.symbol,
                            "name": h.security.security_name,
                            "value_percentage": (float(h.total_value) / total_value * 100) if total_value > 0 else 0
                        }
                        for h in sorted(portfolio.holdings, key=lambda x: x.total_value, reverse=True)[:3]
                    ]
                }
                analysis["portfolio_analysis"].append(portfolio_analysis)
            
            return analysis
            
    except Exception as e:
        logger.error(f"Error analyzing client portfolio: {e}")
        return {"error": f"Failed to analyze portfolio: {str(e)}"}


# List of all portfolio tools for easy import
@tool
async def get_selected_client_portfolios(config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Get portfolio information for the currently selected client.
    This tool automatically uses the client selected in the UI.
    
    Returns:
        Dictionary containing selected client's portfolios, holdings, and performance data
    """
    client_id = get_selected_client_id(config)
    if not client_id:
        return {
            "error": "No client selected. Please select a client in the UI or use get_client_portfolios with a specific client_id."
        }
    
    return await get_client_portfolios.ainvoke({"client_id": client_id})


@tool
async def get_selected_client_transactions(limit: int = 10, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Get transaction history for the currently selected client.
    This tool automatically uses the client selected in the UI.
    
    Args:
        limit: Maximum number of transactions to return (default: 10)
        
    Returns:
        Dictionary containing selected client's recent transactions
    """
    client_id = get_selected_client_id(config)
    if not client_id:
        return {
            "error": "No client selected. Please select a client in the UI or use get_client_transactions with a specific client_id."
        }
    
    return await get_client_transactions.ainvoke({"client_id": client_id, "limit": limit})


@tool
async def analyze_selected_client_performance(config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Perform comprehensive portfolio analysis for the currently selected client.
    This tool automatically uses the client selected in the UI.
    
    Returns:
        Dictionary containing detailed portfolio analysis and performance metrics
    """
    client_id = get_selected_client_id(config)
    if not client_id:
        return {
            "error": "No client selected. Please select a client in the UI or use analyze_client_portfolio_performance with a specific client_id."
        }
    
    return await analyze_client_portfolio_performance.ainvoke({"client_id": client_id})


PORTFOLIO_TOOLS = [
    get_all_clients,
    get_client_portfolios,
    get_client_transactions,
    analyze_client_portfolio_performance,
    # New auto-selected client tools
    get_selected_client_portfolios,
    get_selected_client_transactions,
    analyze_selected_client_performance,
]
