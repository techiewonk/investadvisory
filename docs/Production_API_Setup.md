# Production API Setup Guide

## Overview

The Enhanced Market Research Agent integrates with multiple production APIs to provide real-time market intelligence. This guide explains how to set up and configure these APIs.

## Required API Keys

### 🏛️ **SEC Filings & Company Fundamentals**

- **Primary**: Financial Modeling Prep API
- **Backup**: Alpha Vantage Fundamentals API
- **Purpose**: SEC filings, company profiles, financial ratios, risk analysis
- **Why not direct SEC EDGAR**: Requires authentication and has restrictive rate limits
- **Better alternatives**: Structured, parsed data with additional financial metrics

### 📊 **FRED API (Federal Reserve Economic Data)**

- **Purpose**: Economic indicators (GDP, inflation, unemployment, etc.)
- **Cost**: Free
- **Setup**: Register at https://fred.stlouisfed.org/docs/api/api_key.html
- **Environment Variable**: `FRED_API_KEY=your_key_here`
- **Rate Limits**: 120 requests per minute
- **Documentation**: https://fred.stlouisfed.org/docs/api/

### 📰 **NewsAPI**

- **Purpose**: News sentiment analysis and market psychology
- **Cost**: Free tier (1,000 requests/day), paid plans available
- **Setup**: Register at https://newsapi.org/register
- **Environment Variable**: `NEWS_API_KEY=your_key_here`
- **Rate Limits**: 1,000 requests/day (free), 100 requests/15 minutes
- **Documentation**: https://newsapi.org/docs

### 📈 **Yahoo Finance (yfinance)**

- **Purpose**: Market data, sector performance, technical indicators
- **Cost**: Free
- **Setup**: No API key required, uses yfinance Python library
- **Rate Limits**: Reasonable usage (built-in delays included)
- **Documentation**: https://pypi.org/project/yfinance/

## Optional API Keys

### 🔢 **Alpha Vantage**

- **Purpose**: Advanced market data and technical indicators
- **Cost**: Free tier (5 requests/minute), paid plans available
- **Setup**: Register at https://www.alphavantage.co/support/#api-key
- **Environment Variable**: `ALPHA_VANTAGE_API_KEY=your_key_here`
- **Rate Limits**: 5 requests/minute (free), 75 requests/minute (premium)
- **Documentation**: https://www.alphavantage.co/documentation/

### 💰 **Financial Modeling Prep (Primary SEC Alternative)**

- **Purpose**: SEC filings, company profiles, financial ratios, comprehensive fundamental data
- **Cost**: Free tier (250 requests/day), Starter ($14/month for 10K requests/day)
- **Setup**: Register at https://financialmodelingprep.com/developer/docs
- **Environment Variable**: `FINANCIAL_MODELING_PREP_API_KEY=your_key_here`
- **Rate Limits**: 250 requests/day (free), 10,000/day (starter)
- **Features**:
  - ✅ SEC filings (10-K, 10-Q, 8-K) with direct links
  - ✅ Company profiles and descriptions
  - ✅ Financial ratios and metrics
  - ✅ Historical financial statements
  - ✅ Real-time and historical stock data
- **Documentation**: https://financialmodelingprep.com/developer/docs

### ☁️ **IEX Cloud**

- **Purpose**: Real-time and historical market data
- **Cost**: Free tier (500,000 core data credits/month), paid plans available
- **Setup**: Register at https://iexcloud.io/pricing/
- **Environment Variable**: `IEX_CLOUD_API_KEY=your_key_here`
- **Rate Limits**: Based on credit system
- **Documentation**: https://iexcloud.io/docs/api/

### 🌤️ **OpenWeatherMap**

- **Purpose**: Weather data for sector-specific analysis (agriculture, energy)
- **Cost**: Free tier (1,000 calls/day), paid plans available
- **Setup**: Register at https://openweathermap.org/api
- **Environment Variable**: `OPENWEATHERMAP_API_KEY=your_key_here`
- **Rate Limits**: 1,000 calls/day (free)
- **Documentation**: https://openweathermap.org/api

## Environment Configuration

Create a `.env` file in your project root with the following structure:

```bash
# Core LLM API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Market Research API Keys
FRED_API_KEY=your_fred_api_key_here
NEWS_API_KEY=your_newsapi_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINANCIAL_MODELING_PREP_API_KEY=your_fmp_key_here
IEX_CLOUD_API_KEY=your_iex_cloud_key_here
OPENWEATHERMAP_API_KEY=your_openweather_key_here

# Database Configuration
DATABASE_TYPE=sqlite
SQLITE_DB_PATH=checkpoints.db
SEED_PORTFOLIO_DB=true
```

## Fallback Behavior

The system is designed to gracefully handle missing API keys:

- **Missing API Key**: Falls back to mock data with clear indicators
- **API Rate Limits**: Implements retry logic with exponential backoff
- **API Errors**: Logs warnings and uses fallback data
- **Network Issues**: Caches responses and provides offline capabilities

## Production Deployment Checklist

### ✅ **Essential APIs (Free)**

- [ ] FRED API key configured
- [ ] SEC EDGAR access tested (no key required)
- [ ] Yahoo Finance working (no key required)

### ✅ **Enhanced Features**

- [ ] NewsAPI key for sentiment analysis
- [ ] Alpha Vantage key for advanced technical indicators
- [ ] OpenWeatherMap key for sector-specific weather analysis

### ✅ **Premium Features**

- [ ] Financial Modeling Prep for comprehensive financial data
- [ ] IEX Cloud for real-time market data

### ✅ **Monitoring & Limits**

- [ ] API rate limits monitored
- [ ] Error logging configured
- [ ] Fallback data tested
- [ ] Cache expiration configured (5 minutes default)

## API Usage Examples

### Test API Connections

```python
# Test FRED API
from agents.market_research_tools import get_economic_indicators
result = get_economic_indicators(["GDP", "inflation"])
print(result["data_source"])  # Should show "Federal Reserve Economic Data (FRED)"

# Test NewsAPI
from agents.market_research_tools import analyze_news_sentiment
result = analyze_news_sentiment("AAPL", days_back=7)
print(result["data_source"])  # Should show "NewsAPI"

# Test Yahoo Finance
from agents.market_research_tools import get_sector_performance
result = get_sector_performance("Technology")
print(result["data_source"])  # Should show "Yahoo Finance"
```

### Monitor API Status

```python
# Check which APIs are configured
from core.settings import settings

apis_configured = {
    "FRED": bool(settings.FRED_API_KEY),
    "NewsAPI": bool(settings.NEWS_API_KEY),
    "Alpha Vantage": bool(settings.ALPHA_VANTAGE_API_KEY),
    "Financial Modeling Prep": bool(settings.FINANCIAL_MODELING_PREP_API_KEY),
    "IEX Cloud": bool(settings.IEX_CLOUD_API_KEY),
    "OpenWeatherMap": bool(settings.OPENWEATHERMAP_API_KEY)
}

print("Configured APIs:", apis_configured)
```

## Cost Optimization

### Free Tier Limits

- **FRED**: 120 requests/minute (very generous)
- **NewsAPI**: 1,000 requests/day (sufficient for most use cases)
- **Yahoo Finance**: No official limits (be reasonable)
- **SEC EDGAR**: No official limits (be respectful)

### Caching Strategy

- **Economic Indicators**: Cached for 5 minutes (data updates infrequently)
- **News Sentiment**: Cached for 5 minutes (balance freshness vs. costs)
- **Sector Performance**: Cached for 5 minutes (market hours consideration)
- **SEC Filings**: Cached for 1 hour (filings don't change frequently)

### Rate Limiting

- Built-in delays between requests
- Exponential backoff on errors
- Respect API rate limits automatically
- Queue requests during high usage

## Troubleshooting

### Common Issues

1. **"Mock data" in responses**

   - Check if API key is set in environment
   - Verify API key is valid and active
   - Check rate limits haven't been exceeded

2. **API timeout errors**

   - Network connectivity issues
   - API service temporarily down
   - Rate limits exceeded

3. **Invalid API responses**
   - API key expired or invalid
   - API service changes (update library versions)
   - Malformed requests (check parameters)

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your API calls to see detailed logs
```

## Security Best Practices

- Never commit API keys to version control
- Use environment variables for all sensitive data
- Rotate API keys regularly
- Monitor API usage for unusual patterns
- Use least-privilege access when available
- Consider API key encryption for production deployments

The production API integrations provide institutional-quality market research capabilities while maintaining robust fallback mechanisms for reliability.
