"""Advanced mathematical tools for investment advisory.

This module provides specialized mathematical functions for:
- Risk metrics and portfolio analysis
- Statistical analysis and correlations
- Options pricing and Greeks
- Financial ratios and valuation models
- Portfolio optimization mathematics
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.tools import tool

# Constants for financial calculations
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # Default 5% risk-free rate


@tool
def calculate_portfolio_risk_metrics(returns: List[float], benchmark_returns: List[float] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Args:
        returns: List of portfolio returns (as decimals, e.g., 0.05 for 5%)
        benchmark_returns: Optional list of benchmark returns for comparison
    
    Returns:
        Dictionary containing risk metrics including volatility, Sharpe ratio, VaR, etc.
    """
    try:
        if not returns or len(returns) < 2:
            return {"error": "Need at least 2 return observations"}
        
        # Convert to numpy array for calculations
        returns_array = np.array(returns)
        
        # Basic statistics
        mean_return = np.mean(returns_array)
        std_dev = np.std(returns_array, ddof=1)  # Sample standard deviation
        
        # Annualized metrics
        annualized_return = (1 + mean_return) ** TRADING_DAYS_PER_YEAR - 1
        annualized_volatility = std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Sharpe Ratio
        excess_return = mean_return - (RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
        sharpe_ratio = excess_return / std_dev if std_dev != 0 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns_array, 5)
        var_99 = np.percentile(returns_array, 1)
        
        # Conditional Value at Risk (CVaR/Expected Shortfall)
        cvar_95 = np.mean(returns_array[returns_array <= var_95])
        cvar_99 = np.mean(returns_array[returns_array <= var_99])
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Downside deviation (semi-volatility)
        negative_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else 0
        
        # Sortino Ratio
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        risk_metrics = {
            "mean_return": round(mean_return, 6),
            "annualized_return": round(annualized_return, 4),
            "volatility": round(std_dev, 6),
            "annualized_volatility": round(annualized_volatility, 4),
            "sharpe_ratio": round(annualized_sharpe, 4),
            "sortino_ratio": round(sortino_ratio, 4),
            "var_95": round(var_95, 6),
            "var_99": round(var_99, 6),
            "cvar_95": round(cvar_95, 6),
            "cvar_99": round(cvar_99, 6),
            "max_drawdown": round(max_drawdown, 6),
            "downside_deviation": round(downside_deviation, 6),
            "skewness": round(float(np.mean(((returns_array - mean_return) / std_dev) ** 3)), 4),
            "kurtosis": round(float(np.mean(((returns_array - mean_return) / std_dev) ** 4)) - 3, 4)
        }
        
        # Add benchmark comparison if provided
        if benchmark_returns and len(benchmark_returns) == len(returns):
            benchmark_array = np.array(benchmark_returns)
            
            # Beta calculation
            covariance = np.cov(returns_array, benchmark_array)[0, 1]
            benchmark_variance = np.var(benchmark_array, ddof=1)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Alpha calculation
            benchmark_mean = np.mean(benchmark_array)
            alpha = mean_return - (RISK_FREE_RATE / TRADING_DAYS_PER_YEAR + beta * (benchmark_mean - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR))
            
            # Correlation
            correlation = np.corrcoef(returns_array, benchmark_array)[0, 1]
            
            # Information Ratio
            active_returns = returns_array - benchmark_array
            tracking_error = np.std(active_returns, ddof=1)
            information_ratio = np.mean(active_returns) / tracking_error if tracking_error != 0 else 0
            
            risk_metrics.update({
                "beta": round(beta, 4),
                "alpha": round(alpha, 6),
                "correlation": round(correlation, 4),
                "tracking_error": round(tracking_error, 6),
                "information_ratio": round(information_ratio, 4)
            })
        
        return risk_metrics
        
    except Exception as e:
        return {"error": f"Error calculating risk metrics: {str(e)}"}


@tool
def calculate_correlation_matrix(returns_data: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Calculate correlation matrix and covariance matrix for multiple assets.
    
    Args:
        returns_data: Dictionary with asset names as keys and return lists as values
    
    Returns:
        Dictionary containing correlation matrix, covariance matrix, and analysis
    """
    try:
        if len(returns_data) < 2:
            return {"error": "Need at least 2 assets for correlation analysis"}
        
        # Convert to numpy arrays
        asset_names = list(returns_data.keys())
        returns_matrix = np.array([returns_data[asset] for asset in asset_names])
        
        # Calculate correlation and covariance matrices
        correlation_matrix = np.corrcoef(returns_matrix)
        covariance_matrix = np.cov(returns_matrix, ddof=1)
        
        # Convert to dictionaries for easier interpretation
        correlation_dict = {}
        covariance_dict = {}
        
        for i, asset1 in enumerate(asset_names):
            correlation_dict[asset1] = {}
            covariance_dict[asset1] = {}
            for j, asset2 in enumerate(asset_names):
                correlation_dict[asset1][asset2] = round(correlation_matrix[i, j], 4)
                covariance_dict[asset1][asset2] = round(covariance_matrix[i, j], 8)
        
        # Find highest and lowest correlations (excluding self-correlations)
        correlations = []
        for i in range(len(asset_names)):
            for j in range(i + 1, len(asset_names)):
                correlations.append({
                    "assets": f"{asset_names[i]} - {asset_names[j]}",
                    "correlation": correlation_matrix[i, j]
                })
        
        correlations.sort(key=lambda x: x["correlation"], reverse=True)
        
        # Calculate portfolio diversification metrics
        avg_correlation = np.mean([corr["correlation"] for corr in correlations])
        max_correlation = max(correlations, key=lambda x: x["correlation"])
        min_correlation = min(correlations, key=lambda x: x["correlation"])
        
        return {
            "correlation_matrix": correlation_dict,
            "covariance_matrix": covariance_dict,
            "analysis": {
                "average_correlation": round(avg_correlation, 4),
                "highest_correlation": {
                    "assets": max_correlation["assets"],
                    "value": round(max_correlation["correlation"], 4)
                },
                "lowest_correlation": {
                    "assets": min_correlation["assets"],
                    "value": round(min_correlation["correlation"], 4)
                },
                "diversification_score": round(1 - avg_correlation, 4),  # Higher is better diversified
                "total_assets": len(asset_names)
            }
        }
        
    except Exception as e:
        return {"error": f"Error calculating correlation matrix: {str(e)}"}


@tool
def calculate_black_scholes_option_price(
    stock_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call"
) -> Dict[str, Any]:
    """
    Calculate Black-Scholes option price and Greeks.
    
    Args:
        stock_price: Current stock price
        strike_price: Option strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (as decimal)
        volatility: Implied volatility (as decimal)
        option_type: "call" or "put"
    
    Returns:
        Dictionary containing option price and Greeks
    """
    try:
        from scipy.stats import norm

        # Black-Scholes formula components
        d1 = (math.log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        
        # Calculate option price
        if option_type.lower() == "call":
            option_price = stock_price * norm.cdf(d1) - strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:  # put
            option_price = strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
        
        # Calculate Greeks
        delta = norm.cdf(d1) if option_type.lower() == "call" else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (stock_price * volatility * math.sqrt(time_to_expiry))
        theta_call = (-stock_price * norm.pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry)) 
                     - risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        theta_put = (-stock_price * norm.pdf(d1) * volatility / (2 * math.sqrt(time_to_expiry)) 
                    + risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2))
        theta = theta_call if option_type.lower() == "call" else theta_put
        vega = stock_price * norm.pdf(d1) * math.sqrt(time_to_expiry)
        rho_call = strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        rho_put = -strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        rho = rho_call if option_type.lower() == "call" else rho_put
        
        return {
            "option_price": round(option_price, 4),
            "greeks": {
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "theta": round(theta / 365, 4),  # Daily theta
                "vega": round(vega / 100, 4),    # Vega per 1% volatility change
                "rho": round(rho / 100, 4)       # Rho per 1% interest rate change
            },
            "inputs": {
                "stock_price": stock_price,
                "strike_price": strike_price,
                "time_to_expiry": time_to_expiry,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "option_type": option_type
            }
        }
        
    except ImportError:
        return {"error": "scipy library required for Black-Scholes calculations"}
    except Exception as e:
        return {"error": f"Error calculating Black-Scholes price: {str(e)}"}


@tool
def calculate_portfolio_optimization(
    expected_returns: List[float],
    covariance_matrix: List[List[float]],
    risk_aversion: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate optimal portfolio weights using Modern Portfolio Theory.
    
    Args:
        expected_returns: List of expected returns for each asset
        covariance_matrix: Covariance matrix as list of lists
        risk_aversion: Risk aversion parameter (higher = more conservative)
    
    Returns:
        Dictionary containing optimal weights and portfolio metrics
    """
    try:
        import numpy as np
        from numpy.linalg import inv

        # Convert inputs to numpy arrays
        mu = np.array(expected_returns)
        sigma = np.array(covariance_matrix)
        n = len(mu)
        
        # Calculate optimal weights using mean-variance optimization
        ones = np.ones((n, 1))
        sigma_inv = inv(sigma)
        
        # Calculate components for optimal portfolio
        A = np.dot(ones.T, np.dot(sigma_inv, ones))[0, 0]
        B = np.dot(ones.T, np.dot(sigma_inv, mu))[0]
        C = np.dot(mu.T, np.dot(sigma_inv, mu))
        
        # Optimal weights for given risk aversion
        w_opt = np.dot(sigma_inv, mu) / risk_aversion
        w_opt = w_opt / np.sum(w_opt)  # Normalize to sum to 1
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(w_opt, mu)
        portfolio_variance = np.dot(w_opt.T, np.dot(sigma, w_opt))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
        
        # Calculate minimum variance portfolio
        w_min_var = np.dot(sigma_inv, ones) / A
        w_min_var = w_min_var.flatten()
        min_var_return = np.dot(w_min_var, mu)
        min_var_volatility = np.sqrt(np.dot(w_min_var.T, np.dot(sigma, w_min_var)))
        
        return {
            "optimal_weights": [round(float(w), 4) for w in w_opt],
            "portfolio_metrics": {
                "expected_return": round(float(portfolio_return), 4),
                "volatility": round(float(portfolio_volatility), 4),
                "sharpe_ratio": round(float(sharpe_ratio), 4),
                "risk_aversion": risk_aversion
            },
            "minimum_variance_portfolio": {
                "weights": [round(float(w), 4) for w in w_min_var],
                "expected_return": round(float(min_var_return), 4),
                "volatility": round(float(min_var_volatility), 4)
            },
            "efficient_frontier_params": {
                "A": round(float(A), 6),
                "B": round(float(B), 6),
                "C": round(float(C), 6)
            }
        }
        
    except ImportError:
        return {"error": "numpy library required for portfolio optimization"}
    except Exception as e:
        return {"error": f"Error in portfolio optimization: {str(e)}"}


@tool
def calculate_financial_ratios(
    revenue: float,
    net_income: float,
    total_assets: float,
    total_equity: float,
    total_debt: float,
    current_assets: float,
    current_liabilities: float,
    shares_outstanding: float,
    market_price: float
) -> Dict[str, Any]:
    """
    Calculate comprehensive financial ratios for company analysis.
    
    Args:
        revenue: Total revenue
        net_income: Net income
        total_assets: Total assets
        total_equity: Total shareholders' equity
        total_debt: Total debt
        current_assets: Current assets
        current_liabilities: Current liabilities
        shares_outstanding: Number of shares outstanding
        market_price: Current market price per share
    
    Returns:
        Dictionary containing financial ratios organized by category
    """
    try:
        # Profitability Ratios
        profit_margin = net_income / revenue if revenue != 0 else 0
        roe = net_income / total_equity if total_equity != 0 else 0
        roa = net_income / total_assets if total_assets != 0 else 0
        
        # Liquidity Ratios
        current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
        quick_ratio = (current_assets - current_liabilities * 0.3) / current_liabilities if current_liabilities != 0 else 0  # Approximation
        
        # Leverage Ratios
        debt_to_equity = total_debt / total_equity if total_equity != 0 else 0
        debt_to_assets = total_debt / total_assets if total_assets != 0 else 0
        equity_multiplier = total_assets / total_equity if total_equity != 0 else 0
        
        # Market Ratios
        market_cap = shares_outstanding * market_price
        eps = net_income / shares_outstanding if shares_outstanding != 0 else 0
        pe_ratio = market_price / eps if eps != 0 else 0
        price_to_book = market_price / (total_equity / shares_outstanding) if total_equity != 0 and shares_outstanding != 0 else 0
        market_to_book = market_cap / total_equity if total_equity != 0 else 0
        
        # Efficiency Ratios
        asset_turnover = revenue / total_assets if total_assets != 0 else 0
        equity_turnover = revenue / total_equity if total_equity != 0 else 0
        
        return {
            "profitability_ratios": {
                "profit_margin": round(profit_margin, 4),
                "return_on_equity": round(roe, 4),
                "return_on_assets": round(roa, 4)
            },
            "liquidity_ratios": {
                "current_ratio": round(current_ratio, 4),
                "quick_ratio": round(quick_ratio, 4)
            },
            "leverage_ratios": {
                "debt_to_equity": round(debt_to_equity, 4),
                "debt_to_assets": round(debt_to_assets, 4),
                "equity_multiplier": round(equity_multiplier, 4)
            },
            "market_ratios": {
                "market_cap": round(market_cap, 2),
                "earnings_per_share": round(eps, 4),
                "pe_ratio": round(pe_ratio, 4),
                "price_to_book": round(price_to_book, 4),
                "market_to_book": round(market_to_book, 4)
            },
            "efficiency_ratios": {
                "asset_turnover": round(asset_turnover, 4),
                "equity_turnover": round(equity_turnover, 4)
            }
        }
        
    except Exception as e:
        return {"error": f"Error calculating financial ratios: {str(e)}"}


@tool
def calculate_compound_interest(
    principal: float,
    annual_rate: float,
    years: float,
    compounding_frequency: int = 1
) -> Dict[str, Any]:
    """
    Calculate compound interest and future value with detailed breakdown.
    
    Args:
        principal: Initial investment amount
        annual_rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        years: Number of years
        compounding_frequency: Compounding frequency per year (1=annual, 12=monthly, 365=daily)
    
    Returns:
        Dictionary containing compound interest calculations
    """
    try:
        # Compound interest formula: A = P(1 + r/n)^(nt)
        rate_per_period = annual_rate / compounding_frequency
        total_periods = compounding_frequency * years
        
        future_value = principal * (1 + rate_per_period) ** total_periods
        total_interest = future_value - principal
        
        # Calculate effective annual rate
        effective_rate = (1 + rate_per_period) ** compounding_frequency - 1
        
        # Calculate continuous compounding for comparison
        continuous_fv = principal * math.exp(annual_rate * years)
        
        return {
            "principal": round(principal, 2),
            "annual_rate": round(annual_rate * 100, 2),
            "years": years,
            "compounding_frequency": compounding_frequency,
            "future_value": round(future_value, 2),
            "total_interest": round(total_interest, 2),
            "effective_annual_rate": round(effective_rate * 100, 4),
            "continuous_compounding_fv": round(continuous_fv, 2),
            "growth_multiple": round(future_value / principal, 4)
        }
        
    except Exception as e:
        return {"error": f"Error calculating compound interest: {str(e)}"}


# Collect all advanced math tools
ADVANCED_MATH_TOOLS = [
    calculate_portfolio_risk_metrics,
    calculate_correlation_matrix,
    calculate_black_scholes_option_price,
    calculate_portfolio_optimization,
    calculate_financial_ratios,
    calculate_compound_interest,
]
