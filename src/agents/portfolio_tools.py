"""Portfolio tools for investment advisory agents."""

import logging
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

# Lazy import to avoid circular dependency - imported in functions when needed

logger = logging.getLogger(__name__)


def get_selected_client_id(config: RunnableConfig | None = None) -> str | None:
    """Extract selected client ID from agent config."""
    if not config:
        return None
    
    configurable = getattr(config, "configurable", None)
    if not configurable:
        return None
    
    selected_client = configurable.get("selected_client")
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
        from service.portfolio_service import get_portfolio_service
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
        from service.portfolio_service import get_portfolio_service
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
        from service.portfolio_service import get_portfolio_service
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
        from service.portfolio_service import get_portfolio_service
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
async def get_selected_client_portfolios(
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None
) -> dict[str, Any]:
    """
    Get portfolio information for the currently selected client.
    This tool automatically uses the client selected in the UI.
    
    Returns:
        Dictionary containing selected client's portfolios, holdings, and performance data
    """
    client_id = get_selected_client_id(config)
    if not client_id:
        return {
            "error": "No client selected. Please select a client in the UI or use get_client_portfolios with a specific client_id.",
            "debug_info": {
                "config_received": config is not None,
                "config_type": str(type(config)) if config else None,
                "configurable": getattr(config, "configurable", None) if config else None,
                "selected_client": getattr(config, "configurable", {}).get("selected_client") if config and hasattr(config, "configurable") else None
            }
        }
    
    return await get_client_portfolios.ainvoke({"client_id": client_id})


@tool
async def get_selected_client_transactions(
    limit: int = 10, 
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None
) -> dict[str, Any]:
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
            "error": "No client selected. Please select a client in the UI or use get_client_transactions with a specific client_id.",
            "debug_info": {
                "config_received": config is not None,
                "config_type": str(type(config)) if config else None,
                "configurable": getattr(config, "configurable", None) if config else None,
                "selected_client": getattr(config, "configurable", {}).get("selected_client") if config and hasattr(config, "configurable") else None
            }
        }
    
    return await get_client_transactions.ainvoke({"client_id": client_id, "limit": limit})


@tool
async def analyze_selected_client_performance(
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None
) -> dict[str, Any]:
    """
    Perform comprehensive portfolio analysis for the currently selected client.
    This tool automatically uses the client selected in the UI.
    
    Returns:
        Dictionary containing detailed portfolio analysis and performance metrics
    """
    client_id = get_selected_client_id(config)
    if not client_id:
        return {
            "error": "No client selected. Please select a client in the UI or use analyze_client_portfolio_performance with a specific client_id.",
            "debug_info": {
                "config_received": config is not None,
                "config_type": str(type(config)) if config else None,
                "configurable": getattr(config, "configurable", None) if config else None,
                "selected_client": getattr(config, "configurable", {}).get("selected_client") if config and hasattr(config, "configurable") else None
            }
        }
    
    return await analyze_client_portfolio_performance.ainvoke({"client_id": client_id})


@tool
async def analyze_portfolio_by_market_cap(client_id: str) -> dict[str, Any]:
    """
    Analyze client portfolio breakdown by market capitalization categories.
    
    Args:
        client_id: The client's unique identifier (e.g., 'CLT-001')
        
    Returns:
        Dictionary containing market cap analysis and allocation breakdown
    """
    try:
        from service.portfolio_service import get_portfolio_service
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Market cap categorization based on stock price estimation
            # Note: In production, use real market cap data from financial APIs
            
            market_cap_analysis = {
                "client_id": client_id,
                "total_portfolio_value": 0,
                "market_cap_breakdown": {
                    "large_cap": {"value": 0, "percentage": 0, "holdings": []},
                    "mid_cap": {"value": 0, "percentage": 0, "holdings": []},
                    "small_cap": {"value": 0, "percentage": 0, "holdings": []},
                    "unknown": {"value": 0, "percentage": 0, "holdings": []}
                }
            }
            
            total_value = 0
            for portfolio in result.portfolios:
                total_value += float(portfolio.total_value)
                
                for holding in portfolio.holdings:
                    holding_value = float(holding.total_value)
                    holding_info = {
                        "symbol": holding.security.symbol,
                        "name": holding.security.security_name,
                        "value": holding_value,
                        "quantity": float(holding.total_quantity),
                        "sector": holding.security.sector
                    }
                    
                    # Estimate market cap based on stock price and typical metrics
                    # This is a simplified approach - in production, you'd get real market cap data
                    estimated_market_cap = None
                    if holding.total_quantity > 0:
                        price_per_share = holding_value / float(holding.total_quantity)
                        # Rough estimation - this would need real market cap data in production
                        if price_per_share > 100:  # High price stocks often large cap
                            estimated_market_cap = "large_cap"
                        elif price_per_share > 20:  # Mid-range price
                            estimated_market_cap = "mid_cap"
                        else:  # Lower price stocks
                            estimated_market_cap = "small_cap"
                    
                    # Categorize by estimated market cap
                    category = estimated_market_cap or "unknown"
                    market_cap_analysis["market_cap_breakdown"][category]["value"] += holding_value
                    market_cap_analysis["market_cap_breakdown"][category]["holdings"].append(holding_info)
            
            # Calculate percentages
            market_cap_analysis["total_portfolio_value"] = total_value
            for category in market_cap_analysis["market_cap_breakdown"]:
                category_value = market_cap_analysis["market_cap_breakdown"][category]["value"]
                percentage = (category_value / total_value * 100) if total_value > 0 else 0
                market_cap_analysis["market_cap_breakdown"][category]["percentage"] = round(percentage, 2)
            
            return market_cap_analysis
            
    except Exception as e:
        logger.error(f"Error analyzing portfolio by market cap: {e}")
        return {"error": f"Failed to analyze market cap breakdown: {str(e)}"}


@tool
async def get_individual_stock_performance(client_id: str, symbol: str) -> dict[str, Any]:
    """
    Get detailed performance analysis for a specific stock in a client's portfolio with real-time market data.
    
    Args:
        client_id: The client's unique identifier (e.g., 'CLT-001')
        symbol: Stock symbol to analyze (e.g., 'AAPL')
        
    Returns:
        Dictionary containing detailed stock performance data including:
        - Position summary with real-time current price and YTD performance
        - Holdings breakdown by portfolio
        - Transaction history for the stock
        - Real-time market comparison and year-to-date returns
    """
    try:
        from datetime import datetime

        import yfinance as yf

        from service.portfolio_service import get_portfolio_service
        
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Find the specific stock across all portfolios
            stock_holdings = []
            for portfolio in result.portfolios:
                for holding in portfolio.holdings:
                    if holding.security.symbol.upper() == symbol.upper():
                        stock_holdings.append({
                            "portfolio_name": portfolio.portfolio.name,
                            "symbol": holding.security.symbol,
                            "security_name": holding.security.security_name,
                            "quantity": float(holding.total_quantity),
                            "total_value": float(holding.total_value),
                            "sector": holding.security.sector
                        })
            
            if not stock_holdings:
                return {"error": f"Stock '{symbol}' not found in client's portfolio"}
            
            # Calculate total position across all portfolios
            total_quantity = sum(h["quantity"] for h in stock_holdings)
            
            # Get real-time market data using yfinance
            real_time_data = {}
            ytd_performance = {}
            
            try:
                ticker = yf.Ticker(symbol.upper())
                
                # Get current price
                current_info = ticker.info
                current_price = current_info.get('currentPrice') or current_info.get('regularMarketPrice', 0)
                
                # Get historical data for YTD calculation
                current_year = datetime.now().year
                year_start = f"{current_year}-01-01"
                hist_data = ticker.history(start=year_start, end=datetime.now().strftime('%Y-%m-%d'))
                
                if not hist_data.empty:
                    year_start_price = hist_data['Close'].iloc[0]
                    ytd_return = ((current_price - year_start_price) / year_start_price * 100) if year_start_price > 0 else 0
                    
                    ytd_performance = {
                        "year_start_price": round(float(year_start_price), 2),
                        "current_price": round(float(current_price), 2),
                        "ytd_return_percentage": round(float(ytd_return), 2),
                        "ytd_dollar_change": round(float(current_price - year_start_price), 2)
                    }
                
                real_time_data = {
                    "current_price": round(float(current_price), 2),
                    "market_cap": current_info.get('marketCap'),
                    "pe_ratio": current_info.get('trailingPE'),
                    "52_week_high": current_info.get('fiftyTwoWeekHigh'),
                    "52_week_low": current_info.get('fiftyTwoWeekLow'),
                    "volume": current_info.get('volume'),
                    "avg_volume": current_info.get('averageVolume'),
                    "data_source": "Yahoo Finance (Real-time)"
                }
                
            except Exception as e:
                logger.warning(f"Could not fetch real-time data for {symbol}: {e}")
                # Fallback to portfolio data
                total_value = sum(h["total_value"] for h in stock_holdings)
                current_price = total_value / total_quantity if total_quantity > 0 else 0
                real_time_data = {
                    "current_price": round(current_price, 2),
                    "data_source": "Portfolio data (Historical)"
                }
            
            # Calculate current market value with real-time price
            current_market_value = total_quantity * real_time_data["current_price"]
            
            # Get transaction history for this stock to calculate returns
            transactions_result = await portfolio_service.get_user_transactions(client_id, limit=100, offset=0)
            stock_transactions = []
            
            if transactions_result:
                for t in transactions_result.transactions:
                    if t.security.symbol.upper() == symbol.upper():
                        stock_transactions.append({
                            "date": t.transaction.transaction_date.isoformat(),
                            "type": t.transaction.transaction_type,
                            "quantity": float(t.transaction.quantity),
                            "price": float(t.transaction.price),
                            "total_amount": float(t.transaction.quantity * t.transaction.price)
                        })
            
            # Calculate cost basis and returns
            total_cost = 0
            total_shares_bought = 0
            
            for transaction in stock_transactions:
                if transaction["type"].lower() in ["buy", "purchase"]:
                    total_cost += transaction["total_amount"]
                    total_shares_bought += transaction["quantity"]
            
            average_cost_basis = total_cost / total_shares_bought if total_shares_bought > 0 else 0
            unrealized_pnl = current_market_value - total_cost
            return_percentage = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
            
            # Calculate position-specific YTD performance
            position_ytd_gain_loss = 0
            if ytd_performance:
                position_ytd_gain_loss = total_quantity * ytd_performance["ytd_dollar_change"]
            
            return {
                "client_id": client_id,
                "symbol": symbol.upper(),
                "security_name": stock_holdings[0]["security_name"],
                "sector": stock_holdings[0]["sector"],
                "position_summary": {
                    "total_quantity": total_quantity,
                    "current_price": real_time_data["current_price"],
                    "total_market_value": round(current_market_value, 2),
                    "average_cost_basis": round(average_cost_basis, 2),
                    "total_cost": round(total_cost, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "return_percentage": round(return_percentage, 2),
                    "position_ytd_gain_loss": round(position_ytd_gain_loss, 2)
                },
                "real_time_market_data": real_time_data,
                "ytd_performance": ytd_performance,
                "holdings_by_portfolio": stock_holdings,
                "transaction_history": stock_transactions,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting individual stock performance: {e}")
        return {"error": f"Failed to analyze stock performance: {str(e)}"}


@tool
async def get_best_ytd_performers(client_id: str, limit: int = 5) -> dict[str, Any]:
    """
    Get the best year-to-date performing holdings for a client with real-time market data.
    
    Args:
        client_id: The client's unique identifier (e.g., 'CLT-001')
        limit: Number of top performers to return (default: 5)
        
    Returns:
        Dictionary containing ranked list of best YTD performers with real-time data
    """
    try:
        from datetime import datetime

        import yfinance as yf

        from service.portfolio_service import get_portfolio_service
        
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Get all unique holdings across portfolios
            all_holdings = {}
            for portfolio in result.portfolios:
                for holding in portfolio.holdings:
                    symbol = holding.security.symbol.upper()
                    if symbol not in all_holdings:
                        all_holdings[symbol] = {
                            "symbol": symbol,
                            "security_name": holding.security.security_name,
                            "sector": holding.security.sector,
                            "total_quantity": 0,
                            "total_value": 0
                        }
                    all_holdings[symbol]["total_quantity"] += float(holding.total_quantity)
                    all_holdings[symbol]["total_value"] += float(holding.total_value)
            
            # Get YTD performance for each holding
            performers = []
            current_year = datetime.now().year
            year_start = f"{current_year}-01-01"
            
            for symbol, holding_data in all_holdings.items():
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Get current price and YTD data
                    current_info = ticker.info
                    current_price = current_info.get('currentPrice') or current_info.get('regularMarketPrice', 0)
                    
                    hist_data = ticker.history(start=year_start, end=datetime.now().strftime('%Y-%m-%d'))
                    
                    if not hist_data.empty and current_price > 0:
                        year_start_price = hist_data['Close'].iloc[0]
                        ytd_return = ((current_price - year_start_price) / year_start_price * 100) if year_start_price > 0 else 0
                        
                        # Calculate position-specific gains
                        position_ytd_gain = holding_data["total_quantity"] * (current_price - year_start_price)
                        current_market_value = holding_data["total_quantity"] * current_price
                        
                        performers.append({
                            "symbol": symbol,
                            "security_name": holding_data["security_name"],
                            "sector": holding_data["sector"],
                            "total_quantity": holding_data["total_quantity"],
                            "year_start_price": round(float(year_start_price), 2),
                            "current_price": round(float(current_price), 2),
                            "ytd_return_percentage": round(float(ytd_return), 2),
                            "ytd_dollar_change": round(float(current_price - year_start_price), 2),
                            "position_ytd_gain": round(position_ytd_gain, 2),
                            "current_market_value": round(current_market_value, 2),
                            "market_cap": current_info.get('marketCap'),
                            "pe_ratio": current_info.get('trailingPE')
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not fetch YTD data for {symbol}: {e}")
                    # Add with 0% return if data unavailable
                    performers.append({
                        "symbol": symbol,
                        "security_name": holding_data["security_name"],
                        "sector": holding_data["sector"],
                        "total_quantity": holding_data["total_quantity"],
                        "ytd_return_percentage": 0.0,
                        "position_ytd_gain": 0.0,
                        "current_market_value": holding_data["total_value"],
                        "data_unavailable": True
                    })
            
            # Sort by YTD return percentage (descending)
            performers.sort(key=lambda x: x["ytd_return_percentage"], reverse=True)
            
            # Get top performers
            top_performers = performers[:limit]
            
            # Calculate portfolio impact
            total_portfolio_ytd_gain = sum(p["position_ytd_gain"] for p in performers if "position_ytd_gain" in p)
            
            return {
                "client_id": client_id,
                "analysis_date": datetime.now().isoformat(),
                "total_holdings_analyzed": len(performers),
                "top_performers": top_performers,
                "portfolio_ytd_summary": {
                    "total_ytd_gain_loss": round(total_portfolio_ytd_gain, 2),
                    "best_performer": top_performers[0] if top_performers else None,
                    "worst_performer": performers[-1] if performers else None
                },
                "data_source": "Yahoo Finance (Real-time)"
            }
            
    except Exception as e:
        logger.error(f"Error getting best YTD performers: {e}")
        return {"error": f"Failed to analyze YTD performance: {str(e)}"}


PORTFOLIO_TOOLS = [
    get_all_clients,
    get_client_portfolios,
    get_client_transactions,
    analyze_client_portfolio_performance,
    analyze_portfolio_by_market_cap,
    get_individual_stock_performance,
    get_best_ytd_performers,
    # NOTE: InjectedToolArg tools removed - they don't work with create_react_agent
    # Use explicit client_id parameters instead
    # get_selected_client_portfolios,  # Doesn't work - InjectedToolArg issue
    # get_selected_client_transactions,  # Doesn't work - InjectedToolArg issue  
    # analyze_selected_client_performance,  # Doesn't work - InjectedToolArg issue
]
