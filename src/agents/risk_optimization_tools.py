"""Risk Assessment and Portfolio Optimization Tools for Investment Advisory Platform."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf
from fredapi import Fred
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Risk tolerance categories
RISK_TOLERANCE_CATEGORIES = {
    "conservative": {"max_equity": 40, "max_volatility": 12},
    "moderate_conservative": {"max_equity": 60, "max_volatility": 15},
    "moderate": {"max_equity": 80, "max_volatility": 18},
    "moderate_aggressive": {"max_equity": 90, "max_volatility": 22},
    "aggressive": {"max_equity": 100, "max_volatility": 30}
}

# Regulatory compliance standards
REGULATORY_STANDARDS = {
    "single_position_limit": 0.10,  # Max 10% in single position
    "sector_concentration_limit": 0.25,  # Max 25% in single sector
    "minimum_liquidity": 0.05,  # Min 5% in liquid assets
    "maximum_leverage": 2.0,  # Max 2:1 leverage
    "derivatives_limit": 0.15  # Max 15% in derivatives
}


@tool
async def assess_portfolio_risk_profile(client_id: str) -> Dict[str, Any]:
    """
    Comprehensive portfolio risk assessment including VaR, concentration risk, and liquidity analysis.
    
    Args:
        client_id: The client's unique identifier
        
    Returns:
        Dictionary containing comprehensive risk assessment
    """
    try:
        # Lazy import to avoid circular dependency
        from service.portfolio_service import get_portfolio_service
        
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Calculate portfolio metrics
            total_value = 0
            positions = []
            sector_allocation = {}
            
            for portfolio in result.portfolios:
                for holding in portfolio.holdings:
                    position_value = float(holding.total_value)
                    total_value += position_value
                    
                    positions.append({
                        "symbol": holding.security.symbol,
                        "value": position_value,
                        "sector": holding.security.sector
                    })
                    
                    # Sector allocation
                    sector = holding.security.sector
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + position_value
            
            # Calculate concentration risk
            position_concentrations = []
            for pos in positions:
                concentration = pos["value"] / total_value if total_value > 0 else 0
                position_concentrations.append({
                    "symbol": pos["symbol"],
                    "concentration": round(concentration * 100, 2),
                    "value": pos["value"]
                })
            
            # Sort by concentration
            position_concentrations.sort(key=lambda x: x["concentration"], reverse=True)
            
            # Sector concentration
            sector_concentrations = []
            for sector, value in sector_allocation.items():
                concentration = value / total_value if total_value > 0 else 0
                sector_concentrations.append({
                    "sector": sector,
                    "concentration": round(concentration * 100, 2),
                    "value": value
                })
            
            sector_concentrations.sort(key=lambda x: x["concentration"], reverse=True)
            
            # Risk assessment
            max_position_concentration = max([p["concentration"] for p in position_concentrations]) if position_concentrations else 0
            max_sector_concentration = max([s["concentration"] for s in sector_concentrations]) if sector_concentrations else 0
            
            # Liquidity assessment (simplified)
            liquid_assets_pct = 5.0  # Placeholder - would need real liquidity data
            
            # Risk level determination
            risk_level = "Low"
            if max_position_concentration > 15 or max_sector_concentration > 30:
                risk_level = "High"
            elif max_position_concentration > 10 or max_sector_concentration > 25:
                risk_level = "Medium"
            
            return {
                "client_id": client_id,
                "assessment_date": datetime.now().isoformat(),
                "portfolio_value": round(total_value, 2),
                "risk_level": risk_level,
                "concentration_risk": {
                    "max_position_concentration": round(max_position_concentration, 2),
                    "max_sector_concentration": round(max_sector_concentration, 2),
                    "top_positions": position_concentrations[:5],
                    "sector_breakdown": sector_concentrations
                },
                "liquidity_risk": {
                    "liquid_assets_percentage": liquid_assets_pct,
                    "liquidity_score": "Adequate" if liquid_assets_pct >= 5 else "Low"
                },
                "recommendations": [
                    f"Consider reducing position concentration if above 10%" if max_position_concentration > 10 else "Position concentration within acceptable limits",
                    f"Consider sector diversification if above 25%" if max_sector_concentration > 25 else "Sector allocation appears diversified"
                ]
            }
            
    except Exception as e:
        logger.error(f"Error assessing portfolio risk: {e}")
        return {"error": f"Failed to assess portfolio risk: {str(e)}"}


@tool
async def evaluate_client_risk_tolerance(client_id: str) -> Dict[str, Any]:
    """
    Evaluate client risk tolerance based on profile and portfolio characteristics.
    
    Args:
        client_id: The client's unique identifier
        
    Returns:
        Dictionary containing risk tolerance evaluation
    """
    try:
        # Lazy import to avoid circular dependency
        from service.portfolio_service import get_portfolio_service
        
        async with get_portfolio_service() as portfolio_service:
            # Get client profile
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Extract client information (simplified - would need real client profile data)
            client_profile = result.portfolios[0].user if result.portfolios else None
            
            # Risk tolerance scoring (simplified algorithm)
            risk_score = 3  # Default moderate
            risk_factors = []
            
            if client_profile:
                # Age factor (placeholder - would need real age data)
                age = getattr(client_profile, 'age', 45)  # Default age
                if age < 30:
                    risk_score += 1
                    risk_factors.append("Young age supports higher risk tolerance")
                elif age > 60:
                    risk_score -= 1
                    risk_factors.append("Older age suggests lower risk tolerance")
                
                # Investment horizon (placeholder)
                investment_horizon = getattr(client_profile, 'investment_horizon', 10)  # Default 10 years
                if investment_horizon > 15:
                    risk_score += 1
                    risk_factors.append("Long investment horizon supports higher risk")
                elif investment_horizon < 5:
                    risk_score -= 1
                    risk_factors.append("Short investment horizon suggests lower risk")
            
            # Determine risk category
            risk_categories = ["conservative", "moderate_conservative", "moderate", "moderate_aggressive", "aggressive"]
            risk_category = risk_categories[min(max(risk_score - 1, 0), 4)]
            
            tolerance_data = RISK_TOLERANCE_CATEGORIES[risk_category]
            
            return {
                "client_id": client_id,
                "assessment_date": datetime.now().isoformat(),
                "risk_tolerance": {
                    "category": risk_category.replace("_", " ").title(),
                    "score": risk_score,
                    "max_equity_allocation": tolerance_data["max_equity"],
                    "max_portfolio_volatility": tolerance_data["max_volatility"]
                },
                "risk_factors": risk_factors,
                "recommended_allocation": {
                    "equity": f"{tolerance_data['max_equity']}%",
                    "fixed_income": f"{100 - tolerance_data['max_equity']}%",
                    "alternatives": "5-10%" if risk_score >= 3 else "0-5%"
                },
                "suitable_investments": [
                    "Large-cap stocks" if tolerance_data["max_equity"] >= 60 else "Conservative stocks",
                    "Government bonds",
                    "Corporate bonds" if risk_score >= 2 else "High-grade corporate bonds",
                    "International diversification" if risk_score >= 3 else "Domestic focus"
                ]
            }
            
    except Exception as e:
        logger.error(f"Error evaluating risk tolerance: {e}")
        return {"error": f"Failed to evaluate risk tolerance: {str(e)}"}


@tool
async def check_regulatory_compliance(client_id: str) -> Dict[str, Any]:
    """
    Check portfolio compliance with regulatory requirements and investment guidelines.
    
    Args:
        client_id: The client's unique identifier
        
    Returns:
        Dictionary containing compliance analysis and violations
    """
    try:
        # Lazy import to avoid circular dependency
        from service.portfolio_service import get_portfolio_service
        
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Calculate portfolio metrics for compliance
            total_value = 0
            positions = []
            sector_allocation = {}
            
            for portfolio in result.portfolios:
                for holding in portfolio.holdings:
                    position_value = float(holding.total_value)
                    total_value += position_value
                    
                    positions.append({
                        "symbol": holding.security.symbol,
                        "value": position_value,
                        "sector": holding.security.sector
                    })
                    
                    sector = holding.security.sector
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + position_value
            
            # Check compliance violations
            violations = []
            warnings = []
            
            # Single position limit check
            for pos in positions:
                concentration = pos["value"] / total_value if total_value > 0 else 0
                if concentration > REGULATORY_STANDARDS["single_position_limit"]:
                    violations.append({
                        "type": "Position Concentration",
                        "symbol": pos["symbol"],
                        "current": f"{concentration * 100:.2f}%",
                        "limit": f"{REGULATORY_STANDARDS['single_position_limit'] * 100:.0f}%",
                        "severity": "High"
                    })
                elif concentration > REGULATORY_STANDARDS["single_position_limit"] * 0.8:
                    warnings.append({
                        "type": "Position Concentration Warning",
                        "symbol": pos["symbol"],
                        "current": f"{concentration * 100:.2f}%",
                        "limit": f"{REGULATORY_STANDARDS['single_position_limit'] * 100:.0f}%"
                    })
            
            # Sector concentration check
            for sector, value in sector_allocation.items():
                concentration = value / total_value if total_value > 0 else 0
                if concentration > REGULATORY_STANDARDS["sector_concentration_limit"]:
                    violations.append({
                        "type": "Sector Concentration",
                        "sector": sector,
                        "current": f"{concentration * 100:.2f}%",
                        "limit": f"{REGULATORY_STANDARDS['sector_concentration_limit'] * 100:.0f}%",
                        "severity": "Medium"
                    })
            
            # Compliance status
            compliance_status = "Compliant" if not violations else "Non-Compliant"
            
            return {
                "client_id": client_id,
                "compliance_check_date": datetime.now().isoformat(),
                "compliance_status": compliance_status,
                "violations": violations,
                "warnings": warnings,
                "regulatory_standards": {
                    "single_position_limit": f"{REGULATORY_STANDARDS['single_position_limit'] * 100:.0f}%",
                    "sector_concentration_limit": f"{REGULATORY_STANDARDS['sector_concentration_limit'] * 100:.0f}%",
                    "minimum_liquidity": f"{REGULATORY_STANDARDS['minimum_liquidity'] * 100:.0f}%"
                },
                "recommendations": [
                    f"Reduce {v['symbol']} position to below {v['limit']}" for v in violations if v['type'] == 'Position Concentration'
                ] + [
                    f"Diversify {v['sector']} sector allocation below {v['limit']}" for v in violations if v['type'] == 'Sector Concentration'
                ]
            }
            
    except Exception as e:
        logger.error(f"Error checking compliance: {e}")
        return {"error": f"Failed to check regulatory compliance: {str(e)}"}


@tool
async def generate_portfolio_optimization_recommendations(client_id: str) -> Dict[str, Any]:
    """
    Generate portfolio optimization and rebalancing recommendations based on current market conditions.
    
    Args:
        client_id: The client's unique identifier
        
    Returns:
        Dictionary containing optimization recommendations
    """
    try:
        # Lazy import to avoid circular dependency
        from service.portfolio_service import get_portfolio_service
        
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Get current portfolio allocation
            total_value = 0
            current_allocation = {}
            positions = []
            
            for portfolio in result.portfolios:
                for holding in portfolio.holdings:
                    position_value = float(holding.total_value)
                    total_value += position_value
                    
                    sector = holding.security.sector
                    current_allocation[sector] = current_allocation.get(sector, 0) + position_value
                    
                    positions.append({
                        "symbol": holding.security.symbol,
                        "value": position_value,
                        "percentage": 0  # Will calculate below
                    })
            
            # Calculate current percentages
            for pos in positions:
                pos["percentage"] = (pos["value"] / total_value * 100) if total_value > 0 else 0
            
            # Target allocation (simplified - would use MPT in production)
            target_allocation = {
                "Technology": 25,
                "Healthcare": 15,
                "Financial Services": 15,
                "Consumer Goods": 10,
                "Industrial": 10,
                "Energy": 5,
                "Utilities": 5,
                "Real Estate": 5,
                "Government Bonds": 10
            }
            
            # Calculate rebalancing needs
            rebalancing_actions = []
            for sector, target_pct in target_allocation.items():
                current_value = current_allocation.get(sector, 0)
                current_pct = (current_value / total_value * 100) if total_value > 0 else 0
                difference = target_pct - current_pct
                
                if abs(difference) > 5:  # 5% threshold for rebalancing
                    action = "Increase" if difference > 0 else "Decrease"
                    rebalancing_actions.append({
                        "sector": sector,
                        "action": action,
                        "current_allocation": round(current_pct, 2),
                        "target_allocation": target_pct,
                        "difference": round(difference, 2),
                        "dollar_amount": round(abs(difference) * total_value / 100, 2)
                    })
            
            # Market condition assessment (simplified)
            market_condition = "Neutral"  # Would use real market data
            
            return {
                "client_id": client_id,
                "optimization_date": datetime.now().isoformat(),
                "portfolio_value": round(total_value, 2),
                "market_condition": market_condition,
                "current_allocation": {k: round(v/total_value*100, 2) for k, v in current_allocation.items()},
                "target_allocation": target_allocation,
                "rebalancing_needed": len(rebalancing_actions) > 0,
                "rebalancing_actions": rebalancing_actions,
                "expected_benefits": [
                    "Improved diversification",
                    "Risk-adjusted return optimization",
                    "Alignment with target risk profile"
                ] if rebalancing_actions else ["Portfolio is well-balanced"],
                "implementation_priority": "High" if len(rebalancing_actions) > 3 else "Medium" if rebalancing_actions else "Low"
            }
            
    except Exception as e:
        logger.error(f"Error generating optimization recommendations: {e}")
        return {"error": f"Failed to generate optimization recommendations: {str(e)}"}


@tool
def calculate_optimal_portfolio_allocation(
    expected_returns: List[float],
    risk_tolerance: str = "moderate",
    target_return: Optional[float] = None,
    risk_budget: Optional[float] = None,
    investment_constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate optimal portfolio allocation using Modern Portfolio Theory principles.
    
    Args:
        expected_returns: List of expected returns for assets
        risk_tolerance: Risk tolerance level (conservative, moderate, aggressive)
        target_return: Target portfolio return (optional)
        risk_budget: Maximum acceptable risk level (optional)
        investment_constraints: Investment constraints (optional)
        
    Returns:
        Dictionary containing optimal allocation recommendations
    """
    try:
        import numpy as np

        # Simplified MPT calculation (would use full covariance matrix in production)
        num_assets = len(expected_returns)
        
        # Risk tolerance mapping
        risk_multipliers = {
            "conservative": 0.5,
            "moderate": 1.0,
            "aggressive": 1.5
        }
        
        risk_multiplier = risk_multipliers.get(risk_tolerance, 1.0)
        
        # Simple equal-weight starting point, adjusted by expected returns and risk tolerance
        base_weights = np.array([1.0 / num_assets] * num_assets)
        return_adjustments = np.array(expected_returns) * risk_multiplier
        
        # Normalize adjustments
        if np.sum(return_adjustments) > 0:
            return_adjustments = return_adjustments / np.sum(return_adjustments)
            optimal_weights = (base_weights + return_adjustments) / 2
        else:
            optimal_weights = base_weights
        
        # Ensure weights sum to 1
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(optimal_weights * np.array(expected_returns))
        portfolio_risk = np.sqrt(np.sum((optimal_weights * np.array(expected_returns)) ** 2)) * 0.15  # Simplified risk calc
        
        return {
            "optimization_date": datetime.now().isoformat(),
            "risk_tolerance": risk_tolerance,
            "optimal_weights": [round(w, 4) for w in optimal_weights],
            "expected_portfolio_return": round(portfolio_return, 4),
            "expected_portfolio_risk": round(portfolio_risk, 4),
            "sharpe_ratio": round(portfolio_return / portfolio_risk, 4) if portfolio_risk > 0 else 0,
            "allocation_percentages": [round(w * 100, 2) for w in optimal_weights],
            "diversification_score": round(1 - np.sum(optimal_weights ** 2), 4),  # Herfindahl index
            "recommendations": [
                f"Allocate {round(w * 100, 1)}% to asset {i+1}" for i, w in enumerate(optimal_weights) if w > 0.05
            ]
        }
        
    except Exception as e:
        logger.error(f"Error calculating optimal allocation: {e}")
        return {"error": f"Failed to calculate optimal allocation: {str(e)}"}


@tool
def stress_test_portfolio(
    positions: List[Dict[str, Any]],
    scenario: str = "market_crash"
) -> Dict[str, Any]:
    """
    Perform stress testing on portfolio under various market scenarios.
    
    Args:
        positions: List of portfolio positions with symbols and values
        scenario: Stress test scenario (market_crash, interest_rate_spike, recession, etc.)
        
    Returns:
        Dictionary containing stress test results
    """
    try:
        # Stress test scenarios
        scenarios = {
            "market_crash": {"equity_shock": -0.30, "bond_shock": -0.05, "description": "30% equity decline, 5% bond decline"},
            "interest_rate_spike": {"equity_shock": -0.15, "bond_shock": -0.20, "description": "15% equity decline, 20% bond decline"},
            "recession": {"equity_shock": -0.25, "bond_shock": 0.05, "description": "25% equity decline, 5% bond gain"},
            "inflation_surge": {"equity_shock": -0.10, "bond_shock": -0.15, "description": "10% equity decline, 15% bond decline"},
            "currency_crisis": {"equity_shock": -0.20, "bond_shock": -0.10, "description": "20% equity decline, 10% bond decline"}
        }
        
        if scenario not in scenarios:
            return {"error": f"Unknown scenario: {scenario}"}
        
        scenario_shocks = scenarios[scenario]
        
        # Calculate stress test impact
        total_current_value = sum(pos.get("value", 0) for pos in positions)
        stressed_positions = []
        total_stressed_value = 0
        
        for pos in positions:
            current_value = pos.get("value", 0)
            symbol = pos.get("symbol", "Unknown")
            sector = pos.get("sector", "Unknown")
            
            # Apply shock based on asset type (simplified classification)
            if any(keyword in sector.lower() for keyword in ["bond", "fixed", "treasury"]):
                shock = scenario_shocks["bond_shock"]
            else:
                shock = scenario_shocks["equity_shock"]
            
            stressed_value = current_value * (1 + shock)
            loss = current_value - stressed_value
            
            stressed_positions.append({
                "symbol": symbol,
                "sector": sector,
                "current_value": round(current_value, 2),
                "stressed_value": round(stressed_value, 2),
                "loss": round(loss, 2),
                "loss_percentage": round(shock * 100, 2)
            })
            
            total_stressed_value += stressed_value
        
        total_loss = total_current_value - total_stressed_value
        total_loss_percentage = (total_loss / total_current_value * 100) if total_current_value > 0 else 0
        
        # Risk assessment
        risk_level = "Low"
        if total_loss_percentage > 20:
            risk_level = "High"
        elif total_loss_percentage > 10:
            risk_level = "Medium"
        
        return {
            "stress_test_date": datetime.now().isoformat(),
            "scenario": scenario,
            "scenario_description": scenario_shocks["description"],
            "portfolio_summary": {
                "current_value": round(total_current_value, 2),
                "stressed_value": round(total_stressed_value, 2),
                "total_loss": round(total_loss, 2),
                "loss_percentage": round(total_loss_percentage, 2)
            },
            "risk_level": risk_level,
            "position_details": stressed_positions,
            "recommendations": [
                "Consider hedging strategies" if risk_level == "High" else "Portfolio shows resilience",
                "Diversify across asset classes" if total_loss_percentage > 15 else "Diversification appears adequate",
                "Review position sizing" if any(pos["loss_percentage"] < -25 for pos in stressed_positions) else "Position sizing appropriate"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error performing stress test: {e}")
        return {"error": f"Failed to perform stress test: {str(e)}"}


@tool
async def get_real_time_market_data(symbols: List[str]) -> Dict[str, Any]:
    """
    Get real-time market data for risk assessment and optimization.
    
    Args:
        symbols: List of stock symbols to analyze
        
    Returns:
        Dictionary containing real-time market data and risk indicators
    """
    try:
        import time
        
        market_data = {}
        
        for symbol in symbols[:10]:  # Limit to 10 symbols to avoid rate limits
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1mo")
                
                if not hist.empty:
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                    volatility = hist['Close'].pct_change().std() * (252 ** 0.5) * 100  # Annualized volatility
                    
                    market_data[symbol] = {
                        "current_price": round(float(current_price), 2),
                        "market_cap": info.get('marketCap'),
                        "beta": info.get('beta'),
                        "volatility": round(float(volatility), 2),
                        "volume": info.get('volume'),
                        "52_week_high": info.get('fiftyTwoWeekHigh'),
                        "52_week_low": info.get('fiftyTwoWeekLow')
                    }
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Could not fetch data for {symbol}: {e}")
                market_data[symbol] = {"error": f"Data unavailable: {str(e)}"}
        
        # Calculate market risk indicators
        volatilities = [data.get("volatility", 0) for data in market_data.values() if isinstance(data, dict) and "volatility" in data]
        avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
        
        risk_environment = "Low"
        if avg_volatility > 25:
            risk_environment = "High"
        elif avg_volatility > 15:
            risk_environment = "Medium"
        
        return {
            "data_timestamp": datetime.now().isoformat(),
            "symbols_analyzed": len(market_data),
            "market_data": market_data,
            "risk_indicators": {
                "average_volatility": round(avg_volatility, 2),
                "risk_environment": risk_environment,
                "high_volatility_count": len([v for v in volatilities if v > 30])
            },
            "data_source": "Yahoo Finance"
        }
        
    except Exception as e:
        logger.error(f"Error fetching real-time market data: {e}")
        return {"error": f"Failed to fetch market data: {str(e)}"}


@tool
async def get_economic_risk_indicators() -> Dict[str, Any]:
    """
    Get current economic indicators for risk assessment.
    
    Returns:
        Dictionary containing economic risk indicators
    """
    try:
        from core.settings import settings
        
        economic_data = {}
        
        if settings.FRED_API_KEY:
            try:
                fred = Fred(api_key=settings.FRED_API_KEY.get_secret_value())
                
                # Get key economic indicators
                indicators = {
                    "unemployment_rate": "UNRATE",
                    "inflation_rate": "CPIAUCSL",
                    "federal_funds_rate": "FEDFUNDS",
                    "gdp_growth": "GDP"
                }
                
                for name, series_id in indicators.items():
                    try:
                        data = fred.get_series(series_id, limit=1)
                        if not data.empty:
                            economic_data[name] = {
                                "value": round(float(data.iloc[-1]), 2),
                                "date": data.index[-1].strftime("%Y-%m-%d")
                            }
                        time.sleep(0.2)  # Rate limiting
                    except Exception as e:
                        logger.warning(f"Could not fetch {name}: {e}")
                        economic_data[name] = {"error": str(e)}
                
            except Exception as e:
                logger.warning(f"FRED API error: {e}")
                economic_data["fred_error"] = str(e)
        
        # Get VIX (volatility index) if available
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="1d")
            if not vix_hist.empty:
                vix_value = float(vix_hist['Close'].iloc[-1])
                economic_data["vix"] = {
                    "value": round(vix_value, 2),
                    "interpretation": "High" if vix_value > 30 else "Medium" if vix_value > 20 else "Low"
                }
        except Exception as e:
            logger.warning(f"Could not fetch VIX: {e}")
        
        # Risk assessment based on economic indicators
        risk_factors = []
        if economic_data.get("unemployment_rate", {}).get("value", 0) > 6:
            risk_factors.append("High unemployment rate")
        if economic_data.get("inflation_rate", {}).get("value", 0) > 4:
            risk_factors.append("Elevated inflation")
        if economic_data.get("vix", {}).get("value", 0) > 25:
            risk_factors.append("High market volatility")
        
        economic_risk_level = "High" if len(risk_factors) >= 2 else "Medium" if risk_factors else "Low"
        
        return {
            "data_timestamp": datetime.now().isoformat(),
            "economic_indicators": economic_data,
            "risk_assessment": {
                "economic_risk_level": economic_risk_level,
                "risk_factors": risk_factors,
                "recommendation": "Increase defensive positioning" if economic_risk_level == "High" else "Maintain current allocation"
            },
            "data_sources": ["FRED", "Yahoo Finance"]
        }
        
    except Exception as e:
        logger.error(f"Error fetching economic indicators: {e}")
        return {"error": f"Failed to fetch economic indicators: {str(e)}"}


@tool
async def compare_portfolio_to_market_benchmarks(client_id: str) -> Dict[str, Any]:
    """
    Compare portfolio performance to market benchmarks.
    
    Args:
        client_id: The client's unique identifier
        
    Returns:
        Dictionary containing benchmark comparison analysis
    """
    try:
        # Lazy import to avoid circular dependency
        from service.portfolio_service import get_portfolio_service
        
        async with get_portfolio_service() as portfolio_service:
            result = await portfolio_service.get_user_portfolios(client_id)
            if not result:
                return {"error": f"Client with ID '{client_id}' not found"}
            
            # Get portfolio value (simplified - would need historical data for real comparison)
            total_value = 0
            for portfolio in result.portfolios:
                for holding in portfolio.holdings:
                    total_value += float(holding.total_value)
            
            # Get benchmark data (S&P 500)
            try:
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="1y")
                
                if not spy_hist.empty:
                    spy_start = float(spy_hist['Close'].iloc[0])
                    spy_current = float(spy_hist['Close'].iloc[-1])
                    spy_return = ((spy_current - spy_start) / spy_start) * 100
                    
                    # Portfolio return (placeholder - would need historical portfolio data)
                    portfolio_return = 8.5  # Placeholder
                    
                    alpha = portfolio_return - spy_return
                    
                    benchmark_comparison = {
                        "benchmark": "S&P 500 (SPY)",
                        "benchmark_return": round(spy_return, 2),
                        "portfolio_return": round(portfolio_return, 2),
                        "alpha": round(alpha, 2),
                        "outperformance": alpha > 0,
                        "relative_performance": "Outperforming" if alpha > 0 else "Underperforming"
                    }
                else:
                    benchmark_comparison = {"error": "Could not fetch benchmark data"}
                    
            except Exception as e:
                benchmark_comparison = {"error": f"Benchmark comparison failed: {str(e)}"}
            
            return {
                "client_id": client_id,
                "comparison_date": datetime.now().isoformat(),
                "portfolio_value": round(total_value, 2),
                "benchmark_comparison": benchmark_comparison,
                "risk_adjusted_metrics": {
                    "sharpe_ratio": 1.2,  # Placeholder
                    "beta": 0.95,  # Placeholder
                    "information_ratio": 0.8  # Placeholder
                },
                "recommendations": [
                    "Portfolio shows strong alpha generation" if benchmark_comparison.get("alpha", 0) > 2 else "Consider index fund allocation",
                    "Risk-adjusted returns are competitive" if benchmark_comparison.get("alpha", 0) > 0 else "Review risk management"
                ]
            }
            
    except Exception as e:
        logger.error(f"Error comparing to benchmarks: {e}")
        return {"error": f"Failed to compare to benchmarks: {str(e)}"}


# Risk optimization tools list
RISK_OPTIMIZATION_TOOLS = [
    assess_portfolio_risk_profile,
    evaluate_client_risk_tolerance,
    check_regulatory_compliance,
    generate_portfolio_optimization_recommendations,
    calculate_optimal_portfolio_allocation,
    stress_test_portfolio,
    get_real_time_market_data,
    get_economic_risk_indicators,
    compare_portfolio_to_market_benchmarks,
]
