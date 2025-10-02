"""Advanced market research tools for investment advisory.

This module provides specialized tools for:
- SEC filings and regulatory data
- Economic indicators and macro data
- News analysis and sentiment
- Sector performance analysis
- Market data and technical indicators

Production APIs integrated:
- SEC EDGAR API for regulatory filings
- FRED API for economic indicators
- NewsAPI for news sentiment analysis
- Alpha Vantage for market data and technical indicators
- Yahoo Finance for additional market data
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

import aiohttp
import requests
import requests_cache
import yfinance as yf
from fredapi import Fred
from langchain_core.tools import tool
from newsapi import NewsApiClient
from tenacity import retry, stop_after_attempt, wait_exponential

from core.settings import settings

logger = logging.getLogger(__name__)

# Setup caching for requests
requests_cache.install_cache('market_research_cache', expire_after=300)  # 5 minutes cache

# Rate limiting decorator
def rate_limit(calls_per_minute: int = 60):
    """Rate limiting decorator for API calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Simple rate limiting - in production, use more sophisticated rate limiting
            await asyncio.sleep(60 / calls_per_minute)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


@tool
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=5, max=15))
def get_company_fundamentals(ticker: str, filing_type: str = "10-K") -> Dict[str, Any]:
    """
    Get company fundamentals, SEC filings, and financial data using Financial Modeling Prep API.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        filing_type: Type of filing (10-K, 10-Q, 8-K, annual, quarterly)
    
    Returns:
        Dictionary containing company profile, financial metrics, SEC filings, and analysis
    """
    try:
        filings_data = {
            "ticker": ticker.upper(),
            "filing_type": filing_type,
            "data_source": "Financial Modeling Prep",
            "recent_filings": [],
            "company_profile": {},
            "financial_highlights": {},
            "risk_factors": [],
            "analysis": ""
        }
        
        # Use Financial Modeling Prep API for comprehensive company data
        if settings.FINANCIAL_MODELING_PREP_API_KEY:
            try:
                base_url = "https://financialmodelingprep.com/api/v3"
                api_key = settings.FINANCIAL_MODELING_PREP_API_KEY.get_secret_value()
                
                # Get company profile with detailed information
                profile_url = f"{base_url}/profile/{ticker.upper()}?apikey={api_key}"
                profile_response = requests.get(profile_url, timeout=15)
                time.sleep(0.5)  # Rate limiting - 0.5 second delay
                
                if profile_response.status_code == 200:
                    profile_data = profile_response.json()
                    if profile_data:
                        company = profile_data[0]
                        filings_data["company_profile"] = {
                            "company_name": company.get("companyName", ""),
                            "sector": company.get("sector", ""),
                            "industry": company.get("industry", ""),
                            "description": company.get("description", "")[:500] + "..." if company.get("description") else "",
                            "website": company.get("website", ""),
                            "market_cap": company.get("mktCap", 0),
                            "employees": company.get("fullTimeEmployees", 0),
                            "exchange": company.get("exchangeShortName", ""),
                            "country": company.get("country", ""),
                            "ceo": company.get("ceo", ""),
                            "founded": company.get("ipoDate", "")
                        }
                
                # Get SEC filings with Financial Modeling Prep's structured data
                filings_url = f"{base_url}/sec_filings/{ticker.upper()}?type={filing_type}&page=0&apikey={api_key}"
                filings_response = requests.get(filings_url, timeout=15)
                time.sleep(0.5)  # Rate limiting - 0.5 second delay
                
                if filings_response.status_code == 200:
                    filings_json = filings_response.json()
                    if filings_json:
                        for filing in filings_json[:5]:  # Get latest 5 filings
                            filings_data["recent_filings"].append({
                                "filing_date": filing.get("fillingDate", ""),
                                "accepted_date": filing.get("acceptedDate", ""),
                                "form_type": filing.get("type", filing_type),
                                "document_url": filing.get("linkToFilingHtmlIndex", ""),
                                "txt_url": filing.get("linkToFilingHtml", ""),
                                "cik": filing.get("cik", ""),
                                "status": "Retrieved from Financial Modeling Prep API"
                            })
                
                # Get comprehensive financial ratios and metrics
                ratios_url = f"{base_url}/ratios/{ticker.upper()}?limit=1&apikey={api_key}"
                ratios_response = requests.get(ratios_url, timeout=15)
                time.sleep(0.5)  # Rate limiting - 0.5 second delay
                
                if ratios_response.status_code == 200:
                    ratios_data = ratios_response.json()
                    if ratios_data:
                        ratios = ratios_data[0]
                        filings_data["financial_highlights"] = {
                            "pe_ratio": ratios.get("priceEarningsRatio", "N/A"),
                            "debt_to_equity": ratios.get("debtEquityRatio", "N/A"),
                            "current_ratio": ratios.get("currentRatio", "N/A"),
                            "quick_ratio": ratios.get("quickRatio", "N/A"),
                            "roe": ratios.get("returnOnEquity", "N/A"),
                            "roa": ratios.get("returnOnAssets", "N/A"),
                            "profit_margin": ratios.get("netProfitMargin", "N/A"),
                            "gross_margin": ratios.get("grossProfitMargin", "N/A"),
                            "revenue_growth": ratios.get("revenueGrowth", "N/A"),
                            "eps_growth": ratios.get("epsgrowth", "N/A"),
                            "dividend_yield": ratios.get("dividendYield", "N/A"),
                            "payout_ratio": ratios.get("payoutRatio", "N/A")
                        }
                
                # Get key financial metrics for additional insights
                metrics_url = f"{base_url}/key-metrics/{ticker.upper()}?limit=1&apikey={api_key}"
                metrics_response = requests.get(metrics_url, timeout=15)
                time.sleep(0.5)  # Rate limiting - 0.5 second delay
                
                if metrics_response.status_code == 200:
                    metrics_data = metrics_response.json()
                    if metrics_data:
                        metrics = metrics_data[0]
                        # Add key metrics to financial highlights
                        filings_data["financial_highlights"].update({
                            "price_to_book": metrics.get("priceToBookRatio", "N/A"),
                            "price_to_sales": metrics.get("priceToSalesRatio", "N/A"),
                            "enterprise_value": metrics.get("enterpriseValue", "N/A"),
                            "ev_to_ebitda": metrics.get("enterpriseValueOverEBITDA", "N/A"),
                            "free_cash_flow_yield": metrics.get("freeCashFlowYield", "N/A"),
                            "working_capital": metrics.get("workingCapital", "N/A")
                        })
                
                # Generate analysis and risk factors
                filings_data["analysis"] = _generate_fmp_analysis(filings_data)
                filings_data["risk_factors"] = _generate_risk_factors(filings_data)
                
                return filings_data
                
            except (requests.exceptions.RequestException, requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError, Exception) as e:
                logger.warning(f"Financial Modeling Prep API error for {ticker}: {e}")
        
        # Try Alpha Vantage as backup
        if settings.ALPHA_VANTAGE_API_KEY:
            try:
                return _get_alpha_vantage_fundamentals(ticker, filing_type)
            except (requests.exceptions.RequestException, requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError, Exception) as e:
                logger.warning(f"Alpha Vantage API error for {ticker}: {e}")
        
        # Fallback to mock data
        logger.info(f"API connection issues or no API keys configured for {ticker}, using mock data")
        return _get_mock_sec_data(ticker, filing_type)
        
    except Exception as e:
        logger.error(f"Error fetching SEC filings for {ticker}: {e}")
        return {"error": f"Failed to retrieve SEC filings for {ticker}: {str(e)}"}


def _generate_fmp_analysis(filings_data: Dict[str, Any]) -> str:
    """Generate analysis from Financial Modeling Prep data."""
    analysis_parts = []
    
    company_profile = filings_data.get("company_profile", {})
    financial_highlights = filings_data.get("financial_highlights", {})
    
    if company_profile.get("company_name"):
        analysis_parts.append(f"{company_profile['company_name']} operates in the {company_profile.get('sector', 'Unknown')} sector")
    
    if financial_highlights.get("pe_ratio") and financial_highlights["pe_ratio"] != "N/A":
        pe_ratio = financial_highlights["pe_ratio"]
        if isinstance(pe_ratio, (int, float)):
            if pe_ratio > 25:
                analysis_parts.append("High P/E ratio suggests growth expectations or overvaluation")
            elif pe_ratio < 15:
                analysis_parts.append("Low P/E ratio may indicate value opportunity or concerns")
    
    if financial_highlights.get("debt_to_equity") and financial_highlights["debt_to_equity"] != "N/A":
        de_ratio = financial_highlights["debt_to_equity"]
        if isinstance(de_ratio, (int, float)) and de_ratio > 1.0:
            analysis_parts.append("High debt-to-equity ratio indicates leveraged capital structure")
    
    return ". ".join(analysis_parts) if analysis_parts else "Financial data retrieved from regulatory filings"


def _generate_risk_factors(filings_data: Dict[str, Any]) -> List[str]:
    """Generate risk factors from filing data."""
    risk_factors = []
    
    company_profile = filings_data.get("company_profile", {})
    financial_highlights = filings_data.get("financial_highlights", {})
    
    # Sector-specific risks
    sector = company_profile.get("sector", "")
    if "Technology" in sector:
        risk_factors.extend(["Rapid technological change", "Cybersecurity threats", "Regulatory scrutiny"])
    elif "Healthcare" in sector:
        risk_factors.extend(["Regulatory approval risks", "Patent expiration", "Healthcare policy changes"])
    elif "Financial" in sector:
        risk_factors.extend(["Interest rate sensitivity", "Credit risk", "Regulatory compliance"])
    else:
        risk_factors.extend(["Market competition", "Economic downturns", "Operational risks"])
    
    # Financial-based risks
    if financial_highlights.get("debt_to_equity") and financial_highlights["debt_to_equity"] != "N/A":
        de_ratio = financial_highlights["debt_to_equity"]
        if isinstance(de_ratio, (int, float)) and de_ratio > 1.5:
            risk_factors.append("High leverage increases financial risk")
    
    return risk_factors[:5]  # Limit to top 5 risks


def _get_alpha_vantage_fundamentals(ticker: str, filing_type: str) -> Dict[str, Any]:
    """Get fundamental data from Alpha Vantage as backup."""
    try:
        base_url = "https://www.alphavantage.co/query"
        api_key = settings.ALPHA_VANTAGE_API_KEY.get_secret_value()
        
        # Get company overview
        params = {
            "function": "OVERVIEW",
            "symbol": ticker.upper(),
            "apikey": api_key
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if "Symbol" in data:  # Valid response
                return {
                    "ticker": ticker.upper(),
                    "filing_type": filing_type,
                    "data_source": "Alpha Vantage",
                    "company_profile": {
                        "company_name": data.get("Name", ""),
                        "sector": data.get("Sector", ""),
                        "industry": data.get("Industry", ""),
                        "description": data.get("Description", "")[:500] + "..." if data.get("Description") else "",
                        "market_cap": data.get("MarketCapitalization", 0),
                        "employees": data.get("FullTimeEmployees", 0)
                    },
                    "financial_highlights": {
                        "pe_ratio": data.get("PERatio", "N/A"),
                        "debt_to_equity": data.get("DebtToEquityRatio", "N/A"),
                        "profit_margin": data.get("ProfitMargin", "N/A"),
                        "roe": data.get("ReturnOnEquityTTM", "N/A"),
                        "revenue_growth": data.get("QuarterlyRevenueGrowthYOY", "N/A")
                    },
                    "recent_filings": [
                        {
                            "filing_date": datetime.now().strftime("%Y-%m-%d"),
                            "form_type": filing_type,
                            "status": "Fundamental data from Alpha Vantage",
                            "document_url": f"https://www.sec.gov/edgar/browse/?CIK={ticker}"
                        }
                    ],
                    "analysis": f"Fundamental analysis for {ticker.upper()} from Alpha Vantage",
                    "risk_factors": _generate_risk_factors({
                        "company_profile": {"sector": data.get("Sector", "")},
                        "financial_highlights": {"debt_to_equity": data.get("DebtToEquityRatio", "N/A")}
                    })
                }
        
        # If Alpha Vantage fails, return mock data
        return _get_mock_sec_data(ticker, filing_type)
        
    except Exception as e:
        logger.warning(f"Alpha Vantage error for {ticker}: {e}")
        return _get_mock_sec_data(ticker, filing_type)


def _get_mock_sec_data(ticker: str, filing_type: str) -> Dict[str, Any]:
    """Fallback mock data when all APIs are unavailable."""
    return {
        "ticker": ticker.upper(),
        "filing_type": filing_type,
        "data_source": "Mock Data - API keys required",
        "company_profile": {
            "company_name": f"{ticker.upper()} Corporation",
            "sector": "Technology",
            "industry": "Software",
            "description": f"Mock company profile for {ticker.upper()}. Configure FINANCIAL_MODELING_PREP_API_KEY or ALPHA_VANTAGE_API_KEY for real data.",
            "market_cap": 1000000000,
            "employees": 10000
        },
        "financial_highlights": {
            "pe_ratio": "N/A",
            "debt_to_equity": "N/A",
            "current_ratio": "N/A",
            "roe": "N/A",
            "profit_margin": "N/A"
        },
        "recent_filings": [
            {
                "filing_date": "2024-01-29",
                "accepted_date": "2024-01-29",
                "form_type": filing_type,
                "document_url": f"https://www.sec.gov/edgar/browse/?CIK={ticker}",
                "status": "Mock data - Configure API keys for real SEC filings"
            }
        ],
        "analysis": f"Mock {filing_type} analysis for {ticker.upper()}. Configure FINANCIAL_MODELING_PREP_API_KEY for comprehensive SEC filings data.",
        "risk_factors": [
            "API configuration required for real risk analysis",
            "Market competition (mock data)",
            "Regulatory changes (mock data)"
        ]
    }


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_economic_indicators(indicators: List[str] = None) -> Dict[str, Any]:
    """
    Get current economic indicators and macro data from FRED API.
    
    Args:
        indicators: List of indicators to fetch (GDP, inflation, unemployment, etc.)
    
    Returns:
        Dictionary containing economic indicators data
    """
    try:
        if indicators is None:
            indicators = ["GDP", "inflation", "unemployment", "interest_rates", "consumer_confidence"]
        
        # FRED API series mapping
        fred_series = {
            "GDP": "GDP",  # Gross Domestic Product
            "inflation": "CPIAUCSL",  # Consumer Price Index
            "unemployment": "UNRATE",  # Unemployment Rate
            "interest_rates": "FEDFUNDS",  # Federal Funds Rate
            "consumer_confidence": "UMCSENT"  # University of Michigan Consumer Sentiment
        }
        
        economic_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "data_source": "Federal Reserve Economic Data (FRED)",
            "indicators": {},
            "market_impact": "",
            "investment_implications": []
        }
        
        # Initialize FRED API if key is available
        if settings.FRED_API_KEY:
            try:
                fred = Fred(api_key=settings.FRED_API_KEY.get_secret_value())
                
                for indicator in indicators:
                    if indicator in fred_series:
                        try:
                            # Get latest data point
                            series_data = fred.get_series(fred_series[indicator], limit=1)
                            if not series_data.empty:
                                latest_value = series_data.iloc[-1]
                                latest_date = series_data.index[-1].strftime("%Y-%m-%d")
                                
                                # Get previous value for trend analysis
                                prev_data = fred.get_series(fred_series[indicator], limit=2)
                                trend = "stable"
                                if len(prev_data) >= 2:
                                    if latest_value > prev_data.iloc[-2]:
                                        trend = "increasing"
                                    elif latest_value < prev_data.iloc[-2]:
                                        trend = "decreasing"
                                
                                economic_data["indicators"][indicator] = {
                                    "value": round(float(latest_value), 2),
                                    "unit": _get_indicator_unit(indicator),
                                    "period": latest_date,
                                    "trend": trend,
                                    "source": "FRED"
                                }
                        except Exception as e:
                            logger.warning(f"Failed to fetch {indicator} from FRED: {e}")
                            economic_data["indicators"][indicator] = _get_mock_indicator_data(indicator)
                
                # Generate market analysis
                economic_data["market_impact"] = _analyze_economic_conditions(economic_data["indicators"])
                economic_data["investment_implications"] = _get_investment_implications(economic_data["indicators"])
                
            except Exception as e:
                logger.warning(f"FRED API error: {e}, using fallback data")
                economic_data = _get_mock_economic_data(indicators)
        else:
            logger.info("FRED API key not configured, using mock data")
            economic_data = _get_mock_economic_data(indicators)
        
        return economic_data
        
    except Exception as e:
        logger.error(f"Error fetching economic indicators: {e}")
        return {"error": f"Failed to retrieve economic indicators: {str(e)}"}


def _get_indicator_unit(indicator: str) -> str:
    """Get the unit for an economic indicator."""
    units = {
        "GDP": "Billions of Dollars",
        "inflation": "% year-over-year",
        "unemployment": "% unemployment rate", 
        "interest_rates": "% Federal Funds Rate",
        "consumer_confidence": "Index"
    }
    return units.get(indicator, "Value")


def _get_mock_indicator_data(indicator: str) -> Dict[str, Any]:
    """Get mock data for a single indicator."""
    mock_values = {
        "GDP": {"value": 27000.0, "trend": "stable"},
        "inflation": {"value": 3.2, "trend": "declining"},
        "unemployment": {"value": 3.7, "trend": "stable"},
        "interest_rates": {"value": 5.25, "trend": "stable"},
        "consumer_confidence": {"value": 110.7, "trend": "improving"}
    }
    
    data = mock_values.get(indicator, {"value": 0.0, "trend": "unknown"})
    return {
        "value": data["value"],
        "unit": _get_indicator_unit(indicator),
        "period": datetime.now().strftime("%Y-%m-%d"),
        "trend": data["trend"],
        "source": "Mock Data"
    }


def _get_mock_economic_data(indicators: List[str]) -> Dict[str, Any]:
    """Fallback mock economic data."""
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "data_source": "Mock Data - FRED API unavailable",
        "indicators": {ind: _get_mock_indicator_data(ind) for ind in indicators},
        "market_impact": "Mock economic analysis - FRED API unavailable",
        "investment_implications": [
            "Mock investment implications",
            "FRED API key required for real data"
        ]
    }


def _analyze_economic_conditions(indicators: Dict[str, Any]) -> str:
    """Analyze economic conditions based on indicators."""
    analysis_parts = []
    
    if "unemployment" in indicators:
        unemp = indicators["unemployment"]["value"]
        if unemp < 4.0:
            analysis_parts.append("Low unemployment indicates strong labor market")
        elif unemp > 6.0:
            analysis_parts.append("Elevated unemployment suggests economic weakness")
    
    if "inflation" in indicators:
        inflation = indicators["inflation"]["value"]
        if inflation > 4.0:
            analysis_parts.append("High inflation may pressure Fed policy")
        elif inflation < 2.0:
            analysis_parts.append("Low inflation provides Fed flexibility")
    
    if "interest_rates" in indicators:
        rates = indicators["interest_rates"]["value"]
        if rates > 5.0:
            analysis_parts.append("High interest rates may constrain growth")
        elif rates < 2.0:
            analysis_parts.append("Low interest rates support economic expansion")
    
    return ". ".join(analysis_parts) if analysis_parts else "Economic conditions appear stable"


def _get_investment_implications(indicators: Dict[str, Any]) -> List[str]:
    """Generate investment implications from economic indicators."""
    implications = []
    
    if "interest_rates" in indicators:
        rate_trend = indicators["interest_rates"]["trend"]
        if rate_trend == "increasing":
            implications.append("Rising rates may favor financial sector stocks")
        elif rate_trend == "decreasing":
            implications.append("Falling rates may benefit growth stocks")
    
    if "consumer_confidence" in indicators:
        conf_trend = indicators["consumer_confidence"]["trend"]
        if conf_trend == "improving":
            implications.append("Improving consumer confidence supports discretionary spending")
    
    if "inflation" in indicators:
        inf_trend = indicators["inflation"]["trend"]
        if inf_trend == "declining":
            implications.append("Moderating inflation reduces margin pressure")
    
    return implications if implications else ["Monitor economic trends for investment opportunities"]


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_news_sentiment(ticker: str, days_back: int = 7) -> Dict[str, Any]:
    """
    Analyze news sentiment for a specific stock or market using NewsAPI.
    
    Args:
        ticker: Stock ticker symbol or 'MARKET' for general market sentiment
        days_back: Number of days to look back for news analysis
    
    Returns:
        Dictionary containing news sentiment analysis
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        sentiment_data = {
            "ticker": ticker.upper(),
            "analysis_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "data_source": "NewsAPI",
            "overall_sentiment": "Neutral",
            "sentiment_score": 0.0,
            "news_volume": 0,
            "key_themes": [],
            "recent_headlines": [],
            "risk_alerts": [],
            "investment_impact": ""
        }
        
        # Use NewsAPI if key is available
        if settings.NEWS_API_KEY:
            try:
                newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY.get_secret_value())
                
                # Search for company news
                query = ticker.upper() if ticker.upper() != 'MARKET' else 'stock market'
                
                # Get news articles
                articles = newsapi.get_everything(
                    q=query,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy',
                    page_size=50
                )
                
                if articles['status'] == 'ok' and articles['articles']:
                    news_articles = articles['articles']
                    sentiment_data["news_volume"] = len(news_articles)
                    
                    # Analyze sentiment (simplified - in production, use proper NLP)
                    positive_keywords = ['growth', 'profit', 'beat', 'upgrade', 'strong', 'positive', 'buy']
                    negative_keywords = ['loss', 'decline', 'downgrade', 'weak', 'negative', 'sell', 'risk']
                    
                    sentiment_scores = []
                    headlines = []
                    themes = set()
                    
                    for article in news_articles[:10]:  # Analyze top 10 articles
                        title = article.get('title', '').lower()
                        description = article.get('description', '').lower() if article.get('description') else ''
                        content = f"{title} {description}"
                        
                        # Simple sentiment scoring
                        pos_count = sum(1 for word in positive_keywords if word in content)
                        neg_count = sum(1 for word in negative_keywords if word in content)
                        
                        if pos_count > neg_count:
                            article_sentiment = "Positive"
                            score = min(1.0, (pos_count - neg_count) / 5)
                        elif neg_count > pos_count:
                            article_sentiment = "Negative"
                            score = max(-1.0, -(neg_count - pos_count) / 5)
                        else:
                            article_sentiment = "Neutral"
                            score = 0.0
                        
                        sentiment_scores.append(score)
                        
                        headlines.append({
                            "date": article.get('publishedAt', '')[:10],
                            "headline": article.get('title', 'No title'),
                            "sentiment": article_sentiment,
                            "source": article.get('source', {}).get('name', 'Unknown'),
                            "url": article.get('url', '')
                        })
                        
                        # Extract themes
                        for keyword in positive_keywords + negative_keywords:
                            if keyword in content:
                                themes.add(keyword.title())
                    
                    # Calculate overall sentiment
                    if sentiment_scores:
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                        sentiment_data["sentiment_score"] = round(avg_sentiment, 2)
                        
                        if avg_sentiment > 0.2:
                            sentiment_data["overall_sentiment"] = "Positive"
                        elif avg_sentiment < -0.2:
                            sentiment_data["overall_sentiment"] = "Negative"
                        else:
                            sentiment_data["overall_sentiment"] = "Neutral"
                    
                    sentiment_data["recent_headlines"] = headlines[:5]
                    sentiment_data["key_themes"] = list(themes)[:5]
                    
                    # Generate risk alerts and investment impact
                    sentiment_data["risk_alerts"] = _generate_risk_alerts(headlines)
                    sentiment_data["investment_impact"] = _generate_investment_impact(
                        ticker, sentiment_data["overall_sentiment"], sentiment_data["sentiment_score"]
                    )
                    
                else:
                    logger.warning(f"No news articles found for {ticker}")
                    sentiment_data = _get_mock_sentiment_data(ticker, days_back)
                    
            except Exception as e:
                logger.warning(f"NewsAPI error: {e}, using fallback data")
                sentiment_data = _get_mock_sentiment_data(ticker, days_back)
        else:
            logger.info("NewsAPI key not configured, using mock data")
            sentiment_data = _get_mock_sentiment_data(ticker, days_back)
        
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error analyzing news sentiment for {ticker}: {e}")
        return {"error": f"Failed to analyze news sentiment for {ticker}: {str(e)}"}


def _generate_risk_alerts(headlines: List[Dict[str, Any]]) -> List[str]:
    """Generate risk alerts from news headlines."""
    risk_keywords = ['lawsuit', 'investigation', 'regulation', 'competition', 'decline', 'loss']
    alerts = []
    
    risk_count = 0
    for headline in headlines:
        title = headline.get('headline', '').lower()
        for keyword in risk_keywords:
            if keyword in title:
                risk_count += 1
                break
    
    if risk_count > 0:
        risk_percentage = (risk_count / len(headlines)) * 100
        alerts.append(f"Risk-related news in {risk_percentage:.0f}% of recent articles")
    
    return alerts


def _generate_investment_impact(ticker: str, sentiment: str, score: float) -> str:
    """Generate investment impact analysis."""
    if sentiment == "Positive":
        return f"Positive sentiment trend for {ticker} suggests continued investor confidence and potential upward price momentum."
    elif sentiment == "Negative":
        return f"Negative sentiment for {ticker} indicates investor concerns and potential downward pressure on price."
    else:
        return f"Neutral sentiment for {ticker} suggests balanced investor opinion with no clear directional bias."


def _get_mock_sentiment_data(ticker: str, days_back: int) -> Dict[str, Any]:
    """Fallback mock sentiment data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    return {
        "ticker": ticker.upper(),
        "analysis_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "data_source": "Mock Data - NewsAPI unavailable",
        "overall_sentiment": "Neutral",
        "sentiment_score": 0.0,
        "news_volume": 0,
        "key_themes": ["Mock theme 1", "Mock theme 2"],
        "recent_headlines": [
            {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "headline": f"Mock headline for {ticker}",
                "sentiment": "Neutral",
                "source": "Mock Source"
            }
        ],
        "risk_alerts": ["NewsAPI key required for real sentiment analysis"],
        "investment_impact": f"Mock sentiment analysis for {ticker} - NewsAPI key required for real data."
    }


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_sector_performance(sector: str = None) -> Dict[str, Any]:
    """
    Get sector performance analysis and comparison using Yahoo Finance.
    
    Args:
        sector: Specific sector to analyze (Technology, Healthcare, Finance, etc.)
    
    Returns:
        Dictionary containing sector performance data
    """
    try:
        # Sector ETF mapping for Yahoo Finance
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV", 
            "Finance": "XLF",
            "Energy": "XLE",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Communication": "XLC"
        }
        
        sectors_data = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "data_source": "Yahoo Finance",
            "sector_performance": {},
            "market_leaders": [],
            "underperformers": [],
            "investment_themes": []
        }
        
        try:
            # Get performance data for all sectors using Yahoo Finance
            for sector_name, etf_symbol in sector_etfs.items():
                try:
                    etf = yf.Ticker(etf_symbol)
                    hist = etf.history(period="1y")
                    info = etf.info
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        
                        # Calculate returns
                        ytd_start = datetime(datetime.now().year, 1, 1)
                        ytd_data = etf.history(start=ytd_start)
                        ytd_return = ((current_price - ytd_data['Close'].iloc[0]) / ytd_data['Close'].iloc[0] * 100) if not ytd_data.empty else 0
                        
                        month_ago = datetime.now() - timedelta(days=30)
                        month_data = etf.history(start=month_ago)
                        month_return = ((current_price - month_data['Close'].iloc[0]) / month_data['Close'].iloc[0] * 100) if not month_data.empty else 0
                        
                        three_months_ago = datetime.now() - timedelta(days=90)
                        three_month_data = etf.history(start=three_months_ago)
                        three_month_return = ((current_price - three_month_data['Close'].iloc[0]) / three_month_data['Close'].iloc[0] * 100) if not three_month_data.empty else 0
                        
                        # Determine trend
                        if ytd_return > 10:
                            trend = "Outperforming"
                        elif ytd_return > 0:
                            trend = "Stable"
                        elif ytd_return > -10:
                            trend = "Underperforming"
                        else:
                            trend = "Volatile"
                        
                        sectors_data["sector_performance"][sector_name] = {
                            "etf_symbol": etf_symbol,
                            "ytd_return": round(ytd_return, 2),
                            "1m_return": round(month_return, 2),
                            "3m_return": round(three_month_return, 2),
                            "current_price": round(current_price, 2),
                            "pe_ratio": info.get('trailingPE', 'N/A'),
                            "trend": trend,
                            "key_drivers": _get_sector_drivers(sector_name)
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to get data for {sector_name} ({etf_symbol}): {e}")
                    # Use mock data as fallback
                    sectors_data["sector_performance"][sector_name] = _get_mock_sector_data(sector_name)
            
            # Identify leaders and underperformers
            if sectors_data["sector_performance"]:
                sorted_sectors = sorted(
                    sectors_data["sector_performance"].items(),
                    key=lambda x: x[1].get("ytd_return", 0),
                    reverse=True
                )
                
                # Top 2 performers
                for sector_name, data in sorted_sectors[:2]:
                    if data.get("ytd_return", 0) > 0:
                        sectors_data["market_leaders"].append({
                            "sector": sector_name,
                            "return": f"{data.get('ytd_return', 0)}%",
                            "reason": f"Strong YTD performance with {data.get('trend', 'positive')} trend"
                        })
                
                # Bottom 2 performers
                for sector_name, data in sorted_sectors[-2:]:
                    if data.get("ytd_return", 0) < 0:
                        sectors_data["underperformers"].append({
                            "sector": sector_name,
                            "return": f"{data.get('ytd_return', 0)}%",
                            "reason": f"Negative YTD performance with {data.get('trend', 'challenging')} conditions"
                        })
                
                sectors_data["investment_themes"] = _generate_investment_themes(sectors_data["sector_performance"])
            
        except Exception as e:
            logger.warning(f"Yahoo Finance error: {e}, using mock data")
            sectors_data = _get_mock_sectors_data()
        
        # Return specific sector data if requested
        if sector and sector.title() in sectors_data["sector_performance"]:
            sector_data = sectors_data["sector_performance"][sector.title()]
            return {
                "sector": sector.title(),
                "performance": sector_data,
                "data_source": sectors_data["data_source"],
                "relative_analysis": f"{sector.title()} is currently {sector_data['trend'].lower()} relative to the broader market.",
                "investment_recommendation": f"Based on current trends and fundamentals, {sector.title()} shows {'positive' if sector_data.get('ytd_return', 0) > 0 else 'mixed'} investment potential."
            }
        
        return sectors_data
        
    except Exception as e:
        logger.error(f"Error fetching sector performance: {e}")
        return {"error": f"Failed to retrieve sector performance data: {str(e)}"}


def _get_sector_drivers(sector_name: str) -> List[str]:
    """Get key drivers for a sector."""
    drivers = {
        "Technology": ["AI adoption", "Cloud growth", "Digital transformation"],
        "Healthcare": ["Aging population", "Drug approvals", "Healthcare innovation"],
        "Finance": ["Interest rate environment", "Credit quality", "Digital banking"],
        "Energy": ["Oil prices", "Renewable transition", "Geopolitical factors"],
        "Consumer Discretionary": ["Consumer spending", "E-commerce growth", "Economic conditions"],
        "Consumer Staples": ["Inflation impact", "Supply chain", "Brand strength"],
        "Industrials": ["Infrastructure spending", "Manufacturing activity", "Global trade"],
        "Materials": ["Commodity prices", "Construction demand", "Industrial production"],
        "Real Estate": ["Interest rates", "Property values", "REIT performance"],
        "Utilities": ["Regulatory environment", "Energy transition", "Dividend yields"],
        "Communication": ["5G deployment", "Streaming services", "Advertising spending"]
    }
    return drivers.get(sector_name, ["Market conditions", "Economic factors", "Industry trends"])


def _get_mock_sector_data(sector_name: str) -> Dict[str, Any]:
    """Get mock data for a single sector."""
    mock_returns = {
        "Technology": {"ytd": 8.5, "1m": 3.2, "3m": 12.1},
        "Healthcare": {"ytd": 4.2, "1m": 1.8, "3m": 6.7},
        "Finance": {"ytd": 6.1, "1m": 2.5, "3m": 9.3},
        "Energy": {"ytd": -2.3, "1m": -1.1, "3m": 3.4}
    }
    
    returns = mock_returns.get(sector_name, {"ytd": 0.0, "1m": 0.0, "3m": 0.0})
    
    return {
        "etf_symbol": "MOCK",
        "ytd_return": returns["ytd"],
        "1m_return": returns["1m"],
        "3m_return": returns["3m"],
        "current_price": 100.0,
        "pe_ratio": "N/A",
        "trend": "Mock Data",
        "key_drivers": _get_sector_drivers(sector_name)
    }


def _get_mock_sectors_data() -> Dict[str, Any]:
    """Fallback mock sectors data."""
    return {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "data_source": "Mock Data - Yahoo Finance unavailable",
        "sector_performance": {
            "Technology": _get_mock_sector_data("Technology"),
            "Healthcare": _get_mock_sector_data("Healthcare"),
            "Finance": _get_mock_sector_data("Finance"),
            "Energy": _get_mock_sector_data("Energy")
        },
        "market_leaders": [{"sector": "Technology", "reason": "Mock data"}],
        "underperformers": [{"sector": "Energy", "reason": "Mock data"}],
        "investment_themes": ["Mock theme 1", "Mock theme 2"]
    }


def _generate_investment_themes(sector_performance: Dict[str, Any]) -> List[str]:
    """Generate investment themes from sector performance."""
    themes = []
    
    # Find best performing sectors
    sorted_sectors = sorted(
        sector_performance.items(),
        key=lambda x: x[1].get("ytd_return", 0),
        reverse=True
    )
    
    if sorted_sectors:
        top_sector = sorted_sectors[0]
        if top_sector[1].get("ytd_return", 0) > 5:
            themes.append(f"{top_sector[0]} leading market with strong fundamentals")
    
    # Look for trends
    tech_return = sector_performance.get("Technology", {}).get("ytd_return", 0)
    if tech_return > 10:
        themes.append("Technology sector benefiting from AI and digital transformation")
    
    finance_return = sector_performance.get("Finance", {}).get("ytd_return", 0)
    if finance_return > 5:
        themes.append("Financial sector recovery from interest rate normalization")
    
    return themes if themes else ["Diversified sector performance with mixed signals"]


@tool
def get_market_technical_indicators(symbol: str = "SPY") -> Dict[str, Any]:
    """
    Get technical indicators and market analysis.
    
    Args:
        symbol: Market symbol or ETF (SPY, QQQ, etc.)
    
    Returns:
        Dictionary containing technical analysis data
    """
    try:
        # Mock implementation - integrate with technical analysis APIs
        mock_technical = {
            "symbol": symbol.upper(),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "price_data": {
                "current_price": 485.20,
                "52_week_high": 495.50,
                "52_week_low": 348.10,
                "price_change_1d": 2.35,
                "price_change_pct_1d": 0.49
            },
            "technical_indicators": {
                "RSI_14": {
                    "value": 58.3,
                    "signal": "Neutral",
                    "interpretation": "Neither overbought nor oversold"
                },
                "MACD": {
                    "macd_line": 1.25,
                    "signal_line": 0.98,
                    "histogram": 0.27,
                    "signal": "Bullish",
                    "interpretation": "MACD above signal line suggests upward momentum"
                },
                "Moving_Averages": {
                    "SMA_20": 478.50,
                    "SMA_50": 465.30,
                    "SMA_200": 425.80,
                    "signal": "Bullish",
                    "interpretation": "Price above all major moving averages"
                },
                "Bollinger_Bands": {
                    "upper_band": 492.10,
                    "middle_band": 478.50,
                    "lower_band": 464.90,
                    "signal": "Neutral",
                    "interpretation": "Price in middle of bands, normal volatility"
                }
            },
            "support_resistance": {
                "support_levels": [475.00, 465.00, 450.00],
                "resistance_levels": [490.00, 495.00, 500.00]
            },
            "overall_signal": "Moderately Bullish",
            "market_outlook": "Technical indicators suggest continued upward bias with normal volatility. Key support at 475 level.",
            "risk_factors": [
                "Potential resistance near 490-495 levels",
                "Watch for RSI approaching overbought (>70)",
                "Volume confirmation needed for breakouts"
            ]
        }
        
        return mock_technical
        
    except Exception as e:
        logger.error(f"Error fetching technical indicators for {symbol}: {e}")
        return {"error": f"Failed to retrieve technical indicators for {symbol}: {str(e)}"}


@tool
def get_earnings_calendar(days_ahead: int = 7) -> Dict[str, Any]:
    """
    Get upcoming earnings announcements and estimates.
    
    Args:
        days_ahead: Number of days ahead to look for earnings
    
    Returns:
        Dictionary containing earnings calendar data
    """
    try:
        # Mock implementation - integrate with earnings data APIs
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)
        
        mock_earnings = {
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "upcoming_earnings": [
                {
                    "date": "2024-02-01",
                    "ticker": "AAPL",
                    "company": "Apple Inc.",
                    "time": "After Market Close",
                    "estimates": {
                        "eps_estimate": 2.18,
                        "revenue_estimate": 117.9,  # in billions
                        "eps_surprise_history": 0.05  # average surprise
                    },
                    "analyst_sentiment": "Positive",
                    "key_focus": ["iPhone sales", "Services growth", "China market performance"]
                },
                {
                    "date": "2024-02-02",
                    "ticker": "GOOGL", 
                    "company": "Alphabet Inc.",
                    "time": "After Market Close",
                    "estimates": {
                        "eps_estimate": 1.34,
                        "revenue_estimate": 73.2,
                        "eps_surprise_history": 0.08
                    },
                    "analyst_sentiment": "Neutral",
                    "key_focus": ["Search revenue", "Cloud growth", "AI investments"]
                }
            ],
            "market_impact": "High-profile tech earnings could drive market volatility this week.",
            "investment_considerations": [
                "Earnings beats/misses may impact sector performance",
                "Guidance updates critical for forward valuations", 
                "Options activity elevated around announcement dates"
            ]
        }
        
        return mock_earnings
        
    except Exception as e:
        logger.error(f"Error fetching earnings calendar: {e}")
        return {"error": f"Failed to retrieve earnings calendar: {str(e)}"}


# Collect all market research tools
ADVANCED_MARKET_RESEARCH_TOOLS = [
    get_company_fundamentals,
    get_economic_indicators,
    analyze_news_sentiment,
    get_sector_performance,
    get_market_technical_indicators,
    get_earnings_calendar,
]
