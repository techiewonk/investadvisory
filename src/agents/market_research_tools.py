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
        
        # Return error if no API keys configured
        logger.error(f"No API keys configured for {ticker} - FINANCIAL_MODELING_PREP_API_KEY or ALPHA_VANTAGE_API_KEY required")
        return {"error": f"API keys required for {ticker} - configure FINANCIAL_MODELING_PREP_API_KEY or ALPHA_VANTAGE_API_KEY"}
        
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
        
        # If Alpha Vantage fails, return error
        return {"error": f"Alpha Vantage API failed for {ticker} - check API key and connection"}
        
    except Exception as e:
        logger.warning(f"Alpha Vantage error for {ticker}: {e}")
        return {"error": f"Alpha Vantage API error for {ticker}: {str(e)}"}




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
                            time.sleep(0.2)  # Rate limiting for FRED API
                            
                            if not series_data.empty:
                                latest_value = series_data.iloc[-1]
                                latest_date = series_data.index[-1].strftime("%Y-%m-%d")
                                
                                # Get previous value for trend analysis
                                prev_data = fred.get_series(fred_series[indicator], limit=2)
                                time.sleep(0.2)  # Rate limiting for FRED API
                                
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
                logger.warning(f"FRED API error: {e}")
                return {"error": f"FRED API error: {str(e)}"}
        else:
            logger.error("FRED API key not configured")
            return {"error": "FRED_API_KEY required for economic indicators"}
        
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
                    return {"error": f"No news articles found for {ticker}"}
                    
            except Exception as e:
                logger.warning(f"NewsAPI error: {e}")
                return {"error": f"NewsAPI error: {str(e)}"}
        else:
            logger.error("NewsAPI key not configured")
            return {"error": "NEWS_API_KEY required for news sentiment analysis"}
        
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
                    # Skip failed sectors instead of using mock data
                    continue
            
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
            logger.warning(f"Yahoo Finance error: {e}")
            return {"error": f"Yahoo Finance error: {str(e)}"}
        
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
    Get technical indicators and market analysis using Alpha Vantage API.
    
    Args:
        symbol: Market symbol or ETF (SPY, QQQ, etc.)
    
    Returns:
        Dictionary containing technical analysis data
    """
    try:
        technical_data = {
            "symbol": symbol.upper(),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "data_source": "Alpha Vantage",
            "price_data": {},
            "technical_indicators": {},
            "support_resistance": {},
            "overall_signal": "Neutral",
            "market_outlook": "",
            "risk_factors": []
        }
        
        # Use Alpha Vantage for technical indicators if API key is available
        if settings.ALPHA_VANTAGE_API_KEY:
            try:
                base_url = "https://www.alphavantage.co/query"
                api_key = settings.ALPHA_VANTAGE_API_KEY.get_secret_value()
                
                # Get daily price data
                price_params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol.upper(),
                    "apikey": api_key
                }
                
                price_response = requests.get(base_url, params=price_params, timeout=15)
                time.sleep(1.0)  # Rate limiting for Alpha Vantage (5 calls per minute)
                
                if price_response.status_code == 200:
                    price_data = price_response.json()
                    
                    if "Time Series (Daily)" in price_data:
                        time_series = price_data["Time Series (Daily)"]
                        dates = sorted(time_series.keys(), reverse=True)
                        
                        if len(dates) >= 2:
                            latest_date = dates[0]
                            prev_date = dates[1]
                            
                            current_price = float(time_series[latest_date]["4. close"])
                            prev_price = float(time_series[prev_date]["4. close"])
                            
                            # Calculate price changes
                            price_change = current_price - prev_price
                            price_change_pct = (price_change / prev_price) * 100
                            
                            # Calculate 52-week high/low
                            prices_52w = [float(time_series[date]["2. high"]) for date in dates[:252] if date in time_series]
                            lows_52w = [float(time_series[date]["3. low"]) for date in dates[:252] if date in time_series]
                            
                            technical_data["price_data"] = {
                                "current_price": round(current_price, 2),
                                "52_week_high": round(max(prices_52w) if prices_52w else current_price, 2),
                                "52_week_low": round(min(lows_52w) if lows_52w else current_price, 2),
                                "price_change_1d": round(price_change, 2),
                                "price_change_pct_1d": round(price_change_pct, 2)
                            }
                
                # Get RSI indicator
                rsi_params = {
                    "function": "RSI",
                    "symbol": symbol.upper(),
                    "interval": "daily",
                    "time_period": "14",
                    "series_type": "close",
                    "apikey": api_key
                }
                
                rsi_response = requests.get(base_url, params=rsi_params, timeout=15)
                time.sleep(1.0)  # Rate limiting
                
                if rsi_response.status_code == 200:
                    rsi_data = rsi_response.json()
                    
                    if "Technical Analysis: RSI" in rsi_data:
                        rsi_series = rsi_data["Technical Analysis: RSI"]
                        latest_rsi_date = max(rsi_series.keys())
                        rsi_value = float(rsi_series[latest_rsi_date]["RSI"])
                        
                        # Interpret RSI
                        if rsi_value > 70:
                            rsi_signal = "Overbought"
                            rsi_interpretation = "RSI above 70 suggests potential selling pressure"
                        elif rsi_value < 30:
                            rsi_signal = "Oversold"
                            rsi_interpretation = "RSI below 30 suggests potential buying opportunity"
                        else:
                            rsi_signal = "Neutral"
                            rsi_interpretation = "RSI in normal range, neither overbought nor oversold"
                        
                        technical_data["technical_indicators"]["RSI_14"] = {
                            "value": round(rsi_value, 1),
                            "signal": rsi_signal,
                            "interpretation": rsi_interpretation
                        }
                
                # Get MACD indicator
                macd_params = {
                    "function": "MACD",
                    "symbol": symbol.upper(),
                    "interval": "daily",
                    "series_type": "close",
                    "apikey": api_key
                }
                
                macd_response = requests.get(base_url, params=macd_params, timeout=15)
                time.sleep(1.0)  # Rate limiting
                
                if macd_response.status_code == 200:
                    macd_data = macd_response.json()
                    
                    if "Technical Analysis: MACD" in macd_data:
                        macd_series = macd_data["Technical Analysis: MACD"]
                        latest_macd_date = max(macd_series.keys())
                        
                        macd_line = float(macd_series[latest_macd_date]["MACD"])
                        signal_line = float(macd_series[latest_macd_date]["MACD_Signal"])
                        histogram = float(macd_series[latest_macd_date]["MACD_Hist"])
                        
                        # Interpret MACD
                        if macd_line > signal_line:
                            macd_signal = "Bullish"
                            macd_interpretation = "MACD above signal line suggests upward momentum"
                        else:
                            macd_signal = "Bearish"
                            macd_interpretation = "MACD below signal line suggests downward momentum"
                        
                        technical_data["technical_indicators"]["MACD"] = {
                            "macd_line": round(macd_line, 3),
                            "signal_line": round(signal_line, 3),
                            "histogram": round(histogram, 3),
                            "signal": macd_signal,
                            "interpretation": macd_interpretation
                        }
                
                # Generate overall analysis
                technical_data["overall_signal"] = _generate_technical_signal(technical_data["technical_indicators"])
                technical_data["market_outlook"] = _generate_market_outlook(technical_data)
                technical_data["risk_factors"] = _generate_technical_risks(technical_data)
                
                return technical_data
                
            except Exception as e:
                logger.warning(f"Alpha Vantage API error for {symbol}: {e}")
                return {"error": f"Alpha Vantage API error for {symbol}: {str(e)}"}
        
        # Return error if API not available
        logger.error(f"Alpha Vantage API not configured for {symbol}")
        return {"error": "ALPHA_VANTAGE_API_KEY required for technical indicators"}
        
    except Exception as e:
        logger.error(f"Error fetching technical indicators for {symbol}: {e}")
        return {"error": f"Failed to retrieve technical indicators for {symbol}: {str(e)}"}


def _generate_technical_signal(indicators: Dict[str, Any]) -> str:
    """Generate overall technical signal from indicators."""
    signals = []
    
    if "RSI_14" in indicators:
        rsi_signal = indicators["RSI_14"]["signal"]
        if rsi_signal == "Overbought":
            signals.append("bearish")
        elif rsi_signal == "Oversold":
            signals.append("bullish")
        else:
            signals.append("neutral")
    
    if "MACD" in indicators:
        macd_signal = indicators["MACD"]["signal"]
        if macd_signal == "Bullish":
            signals.append("bullish")
        else:
            signals.append("bearish")
    
    # Determine overall signal
    bullish_count = signals.count("bullish")
    bearish_count = signals.count("bearish")
    
    if bullish_count > bearish_count:
        return "Bullish"
    elif bearish_count > bullish_count:
        return "Bearish"
    else:
        return "Neutral"


def _generate_market_outlook(technical_data: Dict[str, Any]) -> str:
    """Generate market outlook from technical data."""
    signal = technical_data.get("overall_signal", "Neutral")
    price_data = technical_data.get("price_data", {})
    
    outlook_parts = []
    
    if signal == "Bullish":
        outlook_parts.append("Technical indicators suggest positive momentum")
    elif signal == "Bearish":
        outlook_parts.append("Technical indicators suggest negative momentum")
    else:
        outlook_parts.append("Technical indicators show mixed signals")
    
    # Add price context
    if price_data.get("price_change_pct_1d"):
        change_pct = price_data["price_change_pct_1d"]
        if abs(change_pct) > 2:
            outlook_parts.append(f"Recent volatility with {abs(change_pct):.1f}% daily move")
    
    return ". ".join(outlook_parts) if outlook_parts else "Market conditions appear stable"


def _generate_technical_risks(technical_data: Dict[str, Any]) -> List[str]:
    """Generate risk factors from technical analysis."""
    risks = []
    indicators = technical_data.get("technical_indicators", {})
    
    if "RSI_14" in indicators:
        rsi_value = indicators["RSI_14"]["value"]
        if rsi_value > 65:
            risks.append("RSI approaching overbought territory")
        elif rsi_value < 35:
            risks.append("RSI in oversold territory, potential reversal risk")
    
    # General technical risks
    risks.extend([
        "Technical analysis based on historical patterns",
        "Market sentiment can override technical signals",
        "Volume confirmation important for signal validation"
    ])
    
    return risks[:5]  # Limit to top 5 risks




@tool
def get_earnings_calendar(days_ahead: int = 7) -> Dict[str, Any]:
    """
    Get upcoming earnings announcements and estimates using Financial Modeling Prep API.
    
    Args:
        days_ahead: Number of days ahead to look for earnings
    
    Returns:
        Dictionary containing earnings calendar data
    """
    try:
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)
        
        earnings_data = {
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "data_source": "Financial Modeling Prep",
            "upcoming_earnings": [],
            "market_impact": "",
            "investment_considerations": []
        }
        
        # Use Financial Modeling Prep API for earnings calendar
        if settings.FINANCIAL_MODELING_PREP_API_KEY:
            try:
                base_url = "https://financialmodelingprep.com/api/v3"
                api_key = settings.FINANCIAL_MODELING_PREP_API_KEY.get_secret_value()
                
                # Get earnings calendar for the specified period
                from_date = start_date.strftime('%Y-%m-%d')
                to_date = end_date.strftime('%Y-%m-%d')
                
                earnings_url = f"{base_url}/earning_calendar?from={from_date}&to={to_date}&apikey={api_key}"
                earnings_response = requests.get(earnings_url, timeout=15)
                time.sleep(0.5)  # Rate limiting
                
                if earnings_response.status_code == 200:
                    earnings_json = earnings_response.json()
                    
                    if earnings_json and isinstance(earnings_json, list):
                        # Process earnings data
                        for earning in earnings_json[:20]:  # Limit to top 20 earnings
                            try:
                                # Get additional company info
                                ticker = earning.get("symbol", "")
                                if not ticker:
                                    continue
                                
                                # Determine market impact based on market cap (if available)
                                market_cap = earning.get("marketCap", 0)
                                if market_cap > 100000000000:  # > $100B
                                    impact_level = "High"
                                elif market_cap > 10000000000:  # > $10B
                                    impact_level = "Medium"
                                else:
                                    impact_level = "Low"
                                
                                # Format earnings data
                                earnings_entry = {
                                    "date": earning.get("date", ""),
                                    "ticker": ticker,
                                    "company": earning.get("companyName", ticker),
                                    "time": earning.get("time", "Not specified"),
                                    "estimates": {
                                        "eps_estimate": earning.get("epsEstimated", "N/A"),
                                        "revenue_estimate": earning.get("revenueEstimated", "N/A"),
                                        "eps_actual": earning.get("eps", "N/A") if earning.get("eps") else "Pending"
                                    },
                                    "market_cap": market_cap,
                                    "impact_level": impact_level,
                                    "fiscal_date_ending": earning.get("fiscalDateEnding", ""),
                                    "updated_from_date": earning.get("updatedFromDate", "")
                                }
                                
                                earnings_data["upcoming_earnings"].append(earnings_entry)
                                
                            except Exception as e:
                                logger.warning(f"Error processing earnings entry: {e}")
                                continue
                        
                        # Generate market impact analysis
                        high_impact_count = len([e for e in earnings_data["upcoming_earnings"] if e.get("impact_level") == "High"])
                        
                        if high_impact_count > 5:
                            earnings_data["market_impact"] = f"Heavy earnings week with {high_impact_count} major companies reporting. Expect increased market volatility."
                        elif high_impact_count > 2:
                            earnings_data["market_impact"] = f"Moderate earnings activity with {high_impact_count} large-cap companies reporting."
                        else:
                            earnings_data["market_impact"] = "Light earnings week with limited major company reports."
                        
                        # Generate investment considerations
                        earnings_data["investment_considerations"] = [
                            "Earnings surprises can drive significant stock price movements",
                            "Forward guidance often more important than historical results",
                            "Sector-wide impacts possible from major company reports",
                            "Options volatility typically elevated around earnings dates",
                            "Consider position sizing around high-impact earnings"
                        ]
                        
                        return earnings_data
                
                # If no earnings data found, return empty but valid structure
                earnings_data["market_impact"] = "No major earnings scheduled for the specified period."
                earnings_data["investment_considerations"] = ["Monitor for any last-minute earnings announcements"]
                return earnings_data
                
            except Exception as e:
                logger.warning(f"Financial Modeling Prep API error for earnings: {e}")
                return {"error": f"Financial Modeling Prep API error: {str(e)}"}
        
        # Return error if API not available
        logger.error("Financial Modeling Prep API not configured for earnings")
        return {"error": "FINANCIAL_MODELING_PREP_API_KEY required for earnings calendar"}
        
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
