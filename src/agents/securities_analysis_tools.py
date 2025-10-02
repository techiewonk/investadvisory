"""Advanced Securities Analysis Tools for Technical and Statistical Analysis."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Technical Analysis Helper Functions
def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=window).mean()

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=window).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    
    return {
        'upper': sma + (std * num_std),
        'middle': sma,
        'lower': sma - (std * num_std)
    }

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        'k_percent': k_percent,
        'd_percent': d_percent
    }

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


@tool
async def perform_technical_analysis(symbol: str, period: str = "1y", indicators: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform comprehensive technical analysis on a security with multiple indicators.
    
    Args:
        symbol: Stock symbol to analyze (e.g., 'AAPL', 'MSFT')
        period: Time period for analysis ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        indicators: List of indicators to calculate. If None, calculates all major indicators.
                   Options: ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 'atr', 'obv', 'vwap']
    
    Returns:
        Dictionary containing technical analysis results and interpretations
    """
    try:
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 'atr', 'obv', 'vwap']
        
        # Fetch data
        ticker = yf.Ticker(symbol.upper())
        hist_data = ticker.history(period=period)
        
        if hist_data.empty:
            return {"error": f"No data available for symbol {symbol}"}
        
        # Get current info
        info = ticker.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        
        results = {
            "symbol": symbol.upper(),
            "analysis_date": datetime.now().isoformat(),
            "period_analyzed": period,
            "current_price": round(float(current_price), 2),
            "data_points": len(hist_data),
            "price_range": {
                "high": round(hist_data['High'].max(), 2),
                "low": round(hist_data['Low'].min(), 2),
                "latest": round(hist_data['Close'].iloc[-1], 2)
            },
            "technical_indicators": {},
            "signals": [],
            "interpretation": {}
        }
        
        close = hist_data['Close']
        high = hist_data['High']
        low = hist_data['Low']
        volume = hist_data['Volume']
        
        # Calculate requested indicators
        if 'sma' in indicators:
            sma_20 = calculate_sma(close, 20)
            sma_50 = calculate_sma(close, 50)
            sma_200 = calculate_sma(close, 200)
            
            results["technical_indicators"]["sma"] = {
                "sma_20": round(sma_20.iloc[-1], 2),
                "sma_50": round(sma_50.iloc[-1], 2),
                "sma_200": round(sma_200.iloc[-1], 2)
            }
            
            # SMA signals
            if current_price > sma_20.iloc[-1]:
                results["signals"].append("Price above 20-day SMA (Bullish short-term)")
            if current_price > sma_50.iloc[-1]:
                results["signals"].append("Price above 50-day SMA (Bullish medium-term)")
            if current_price > sma_200.iloc[-1]:
                results["signals"].append("Price above 200-day SMA (Bullish long-term)")
        
        if 'ema' in indicators:
            ema_12 = calculate_ema(close, 12)
            ema_26 = calculate_ema(close, 26)
            
            results["technical_indicators"]["ema"] = {
                "ema_12": round(ema_12.iloc[-1], 2),
                "ema_26": round(ema_26.iloc[-1], 2)
            }
        
        if 'rsi' in indicators:
            rsi = calculate_rsi(close)
            current_rsi = rsi.iloc[-1]
            
            results["technical_indicators"]["rsi"] = {
                "current_rsi": round(current_rsi, 2),
                "interpretation": "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            }
            
            if current_rsi > 70:
                results["signals"].append(f"RSI overbought at {current_rsi:.1f} (Potential sell signal)")
            elif current_rsi < 30:
                results["signals"].append(f"RSI oversold at {current_rsi:.1f} (Potential buy signal)")
        
        if 'macd' in indicators:
            macd_data = calculate_macd(close)
            current_macd = macd_data['macd'].iloc[-1]
            current_signal = macd_data['signal'].iloc[-1]
            current_histogram = macd_data['histogram'].iloc[-1]
            
            results["technical_indicators"]["macd"] = {
                "macd_line": round(current_macd, 4),
                "signal_line": round(current_signal, 4),
                "histogram": round(current_histogram, 4)
            }
            
            if current_macd > current_signal:
                results["signals"].append("MACD above signal line (Bullish momentum)")
            else:
                results["signals"].append("MACD below signal line (Bearish momentum)")
        
        if 'bollinger' in indicators:
            bb = calculate_bollinger_bands(close)
            
            results["technical_indicators"]["bollinger_bands"] = {
                "upper_band": round(bb['upper'].iloc[-1], 2),
                "middle_band": round(bb['middle'].iloc[-1], 2),
                "lower_band": round(bb['lower'].iloc[-1], 2),
                "bandwidth": round((bb['upper'].iloc[-1] - bb['lower'].iloc[-1]) / bb['middle'].iloc[-1] * 100, 2)
            }
            
            if current_price > bb['upper'].iloc[-1]:
                results["signals"].append("Price above upper Bollinger Band (Potential overbought)")
            elif current_price < bb['lower'].iloc[-1]:
                results["signals"].append("Price below lower Bollinger Band (Potential oversold)")
        
        if 'stochastic' in indicators:
            stoch = calculate_stochastic(high, low, close)
            
            results["technical_indicators"]["stochastic"] = {
                "k_percent": round(stoch['k_percent'].iloc[-1], 2),
                "d_percent": round(stoch['d_percent'].iloc[-1], 2)
            }
            
            k_val = stoch['k_percent'].iloc[-1]
            if k_val > 80:
                results["signals"].append(f"Stochastic overbought at {k_val:.1f}")
            elif k_val < 20:
                results["signals"].append(f"Stochastic oversold at {k_val:.1f}")
        
        if 'atr' in indicators:
            atr = calculate_atr(high, low, close)
            
            results["technical_indicators"]["atr"] = {
                "current_atr": round(atr.iloc[-1], 2),
                "volatility_percentage": round(atr.iloc[-1] / current_price * 100, 2)
            }
        
        if 'obv' in indicators:
            obv = calculate_obv(close, volume)
            
            results["technical_indicators"]["obv"] = {
                "current_obv": int(obv.iloc[-1]),
                "trend": "Rising" if obv.iloc[-1] > obv.iloc[-10] else "Falling"
            }
        
        if 'vwap' in indicators:
            vwap = calculate_vwap(high, low, close, volume)
            
            results["technical_indicators"]["vwap"] = {
                "current_vwap": round(vwap.iloc[-1], 2),
                "price_vs_vwap": "Above VWAP" if current_price > vwap.iloc[-1] else "Below VWAP"
            }
        
        # Overall interpretation
        bullish_signals = len([s for s in results["signals"] if "Bullish" in s or "buy" in s.lower()])
        bearish_signals = len([s for s in results["signals"] if "Bearish" in s or "sell" in s.lower()])
        
        if bullish_signals > bearish_signals:
            results["interpretation"]["overall"] = "Bullish"
        elif bearish_signals > bullish_signals:
            results["interpretation"]["overall"] = "Bearish"
        else:
            results["interpretation"]["overall"] = "Neutral"
        
        results["interpretation"]["confidence"] = "High" if abs(bullish_signals - bearish_signals) >= 3 else "Medium" if abs(bullish_signals - bearish_signals) >= 1 else "Low"
        
        return results
        
    except Exception as e:
        logger.error(f"Error in technical analysis for {symbol}: {e}")
        return {"error": f"Failed to perform technical analysis: {str(e)}"}


@tool
async def analyze_security_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Analyze fundamental data for a security including financial ratios and company metrics.
    
    Args:
        symbol: Stock symbol to analyze
        
    Returns:
        Dictionary containing fundamental analysis
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        if not info or 'symbol' not in info:
            return {"error": f"No fundamental data available for {symbol}"}
        
        # Extract key fundamental metrics
        fundamentals = {
            "symbol": symbol.upper(),
            "company_name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "analysis_date": datetime.now().isoformat(),
            
            # Valuation Metrics
            "valuation": {
                "market_cap": info.get('marketCap'),
                "enterprise_value": info.get('enterpriseValue'),
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "peg_ratio": info.get('pegRatio'),
                "price_to_book": info.get('priceToBook'),
                "price_to_sales": info.get('priceToSalesTrailing12Months'),
                "ev_to_revenue": info.get('enterpriseToRevenue'),
                "ev_to_ebitda": info.get('enterpriseToEbitda')
            },
            
            # Profitability Metrics
            "profitability": {
                "profit_margin": info.get('profitMargins'),
                "operating_margin": info.get('operatingMargins'),
                "return_on_assets": info.get('returnOnAssets'),
                "return_on_equity": info.get('returnOnEquity'),
                "revenue_growth": info.get('revenueGrowth'),
                "earnings_growth": info.get('earningsGrowth')
            },
            
            # Financial Health
            "financial_health": {
                "total_cash": info.get('totalCash'),
                "total_debt": info.get('totalDebt'),
                "debt_to_equity": info.get('debtToEquity'),
                "current_ratio": info.get('currentRatio'),
                "quick_ratio": info.get('quickRatio'),
                "free_cash_flow": info.get('freeCashflow')
            },
            
            # Dividend Information
            "dividend": {
                "dividend_yield": info.get('dividendYield'),
                "dividend_rate": info.get('dividendRate'),
                "payout_ratio": info.get('payoutRatio'),
                "ex_dividend_date": info.get('exDividendDate')
            },
            
            # Trading Metrics
            "trading": {
                "beta": info.get('beta'),
                "52_week_high": info.get('fiftyTwoWeekHigh'),
                "52_week_low": info.get('fiftyTwoWeekLow'),
                "avg_volume": info.get('averageVolume'),
                "shares_outstanding": info.get('sharesOutstanding'),
                "float_shares": info.get('floatShares')
            }
        }
        
        # Add interpretation
        interpretation = []
        
        # P/E Ratio interpretation
        pe_ratio = fundamentals["valuation"]["pe_ratio"]
        if pe_ratio:
            if pe_ratio < 15:
                interpretation.append("Low P/E ratio - potentially undervalued")
            elif pe_ratio > 25:
                interpretation.append("High P/E ratio - potentially overvalued or high growth expected")
            else:
                interpretation.append("Moderate P/E ratio - fairly valued")
        
        # Debt-to-Equity interpretation
        debt_to_equity = fundamentals["financial_health"]["debt_to_equity"]
        if debt_to_equity:
            if debt_to_equity > 2.0:
                interpretation.append("High debt-to-equity ratio - potential financial risk")
            elif debt_to_equity < 0.5:
                interpretation.append("Low debt-to-equity ratio - conservative capital structure")
        
        # ROE interpretation
        roe = fundamentals["profitability"]["return_on_equity"]
        if roe:
            if roe > 0.15:
                interpretation.append("Strong return on equity - efficient use of shareholder capital")
            elif roe < 0.05:
                interpretation.append("Weak return on equity - poor capital efficiency")
        
        fundamentals["interpretation"] = interpretation
        
        return fundamentals
        
    except Exception as e:
        logger.error(f"Error in fundamental analysis for {symbol}: {e}")
        return {"error": f"Failed to perform fundamental analysis: {str(e)}"}


@tool
async def compare_securities(symbols: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare multiple securities across various financial and technical metrics.
    
    Args:
        symbols: List of stock symbols to compare
        metrics: List of metrics to compare. If None, uses default set.
                Options: ['price', 'pe_ratio', 'market_cap', 'rsi', 'beta', 'dividend_yield']
    
    Returns:
        Dictionary containing comparison analysis
    """
    try:
        if not symbols or len(symbols) < 2:
            return {"error": "At least 2 symbols required for comparison"}
        
        if len(symbols) > 10:
            return {"error": "Maximum 10 symbols allowed for comparison"}
        
        if metrics is None:
            metrics = ['price', 'pe_ratio', 'market_cap', 'rsi', 'beta', 'dividend_yield']
        
        comparison_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol.upper())
                info = ticker.info
                hist = ticker.history(period="3mo")
                
                if hist.empty:
                    continue
                
                # Calculate RSI for technical comparison
                rsi = calculate_rsi(hist['Close']).iloc[-1] if len(hist) > 14 else None
                
                comparison_data[symbol.upper()] = {
                    'price': info.get('currentPrice') or info.get('regularMarketPrice', 0),
                    'pe_ratio': info.get('trailingPE'),
                    'market_cap': info.get('marketCap'),
                    'beta': info.get('beta'),
                    'dividend_yield': info.get('dividendYield'),
                    'rsi': rsi,
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    '52_week_high': info.get('fiftyTwoWeekHigh'),
                    '52_week_low': info.get('fiftyTwoWeekLow')
                }
                
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {e}")
                continue
        
        if not comparison_data:
            return {"error": "No valid data found for any of the provided symbols"}
        
        # Create comparison analysis
        results = {
            "comparison_date": datetime.now().isoformat(),
            "symbols_analyzed": list(comparison_data.keys()),
            "metrics_compared": metrics,
            "data": comparison_data,
            "rankings": {},
            "analysis": []
        }
        
        # Create rankings for each metric
        for metric in metrics:
            if metric in ['price', 'market_cap', 'dividend_yield']:
                # Higher is better
                sorted_symbols = sorted(
                    [s for s in comparison_data.keys() if comparison_data[s].get(metric) is not None],
                    key=lambda s: comparison_data[s][metric] or 0,
                    reverse=True
                )
            elif metric in ['pe_ratio', 'beta']:
                # Lower is generally better (for P/E and beta close to 1)
                sorted_symbols = sorted(
                    [s for s in comparison_data.keys() if comparison_data[s].get(metric) is not None],
                    key=lambda s: comparison_data[s][metric] or float('inf')
                )
            elif metric == 'rsi':
                # Closer to 50 is better (neutral)
                sorted_symbols = sorted(
                    [s for s in comparison_data.keys() if comparison_data[s].get(metric) is not None],
                    key=lambda s: abs((comparison_data[s][metric] or 50) - 50)
                )
            else:
                sorted_symbols = list(comparison_data.keys())
            
            results["rankings"][metric] = sorted_symbols
        
        # Generate analysis insights
        if 'market_cap' in metrics and len(comparison_data) > 1:
            market_caps = {s: d.get('market_cap') for s, d in comparison_data.items() if d.get('market_cap')}
            if market_caps:
                largest = max(market_caps.items(), key=lambda x: x[1])
                smallest = min(market_caps.items(), key=lambda x: x[1])
                results["analysis"].append(f"{largest[0]} has the largest market cap at ${largest[1]:,.0f}")
                results["analysis"].append(f"{smallest[0]} has the smallest market cap at ${smallest[1]:,.0f}")
        
        if 'pe_ratio' in metrics:
            pe_ratios = {s: d.get('pe_ratio') for s, d in comparison_data.items() if d.get('pe_ratio')}
            if pe_ratios:
                lowest_pe = min(pe_ratios.items(), key=lambda x: x[1])
                results["analysis"].append(f"{lowest_pe[0]} has the lowest P/E ratio at {lowest_pe[1]:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in securities comparison: {e}")
        return {"error": f"Failed to compare securities: {str(e)}"}


@tool
async def analyze_security_patterns(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """
    Identify chart patterns and trends in security price data.
    
    Args:
        symbol: Stock symbol to analyze
        period: Time period for pattern analysis
        
    Returns:
        Dictionary containing pattern analysis results
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        hist_data = ticker.history(period=period)
        
        if hist_data.empty:
            return {"error": f"No data available for {symbol}"}
        
        close = hist_data['Close']
        high = hist_data['High']
        low = hist_data['Low']
        
        patterns = {
            "symbol": symbol.upper(),
            "analysis_date": datetime.now().isoformat(),
            "period_analyzed": period,
            "patterns_detected": [],
            "trend_analysis": {},
            "support_resistance": {},
            "price_action": {}
        }
        
        # Trend Analysis
        sma_20 = calculate_sma(close, 20)
        sma_50 = calculate_sma(close, 50)
        sma_200 = calculate_sma(close, 200)
        
        current_price = close.iloc[-1]
        
        # Determine overall trend
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]:
            trend = "Strong Uptrend"
        elif current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend = "Moderate Uptrend"
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]:
            trend = "Strong Downtrend"
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend = "Moderate Downtrend"
        else:
            trend = "Sideways/Consolidation"
        
        patterns["trend_analysis"] = {
            "overall_trend": trend,
            "short_term": "Bullish" if current_price > sma_20.iloc[-1] else "Bearish",
            "medium_term": "Bullish" if current_price > sma_50.iloc[-1] else "Bearish",
            "long_term": "Bullish" if current_price > sma_200.iloc[-1] else "Bearish"
        }
        
        # Support and Resistance Levels
        recent_high = high.tail(50).max()
        recent_low = low.tail(50).min()
        
        patterns["support_resistance"] = {
            "resistance_level": round(recent_high, 2),
            "support_level": round(recent_low, 2),
            "distance_to_resistance": round((recent_high - current_price) / current_price * 100, 2),
            "distance_to_support": round((current_price - recent_low) / current_price * 100, 2)
        }
        
        # Price Action Analysis
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        patterns["price_action"] = {
            "current_price": round(current_price, 2),
            "volatility_annualized": round(volatility, 2),
            "average_daily_return": round(daily_returns.mean() * 100, 3),
            "max_daily_gain": round(daily_returns.max() * 100, 2),
            "max_daily_loss": round(daily_returns.min() * 100, 2)
        }
        
        # Pattern Detection (Simplified)
        # Golden Cross
        if len(sma_50) > 1 and len(sma_200) > 1:
            if sma_50.iloc[-1] > sma_200.iloc[-1] and sma_50.iloc[-2] <= sma_200.iloc[-2]:
                patterns["patterns_detected"].append("Golden Cross - 50-day MA crossed above 200-day MA (Bullish)")
            elif sma_50.iloc[-1] < sma_200.iloc[-1] and sma_50.iloc[-2] >= sma_200.iloc[-2]:
                patterns["patterns_detected"].append("Death Cross - 50-day MA crossed below 200-day MA (Bearish)")
        
        # Breakout Detection
        bb = calculate_bollinger_bands(close)
        if current_price > bb['upper'].iloc[-1]:
            patterns["patterns_detected"].append("Bollinger Band Breakout - Price above upper band")
        elif current_price < bb['lower'].iloc[-1]:
            patterns["patterns_detected"].append("Bollinger Band Breakdown - Price below lower band")
        
        # Volume Analysis
        if 'Volume' in hist_data.columns:
            avg_volume = hist_data['Volume'].tail(20).mean()
            recent_volume = hist_data['Volume'].iloc[-1]
            
            if recent_volume > avg_volume * 1.5:
                patterns["patterns_detected"].append("High Volume Activity - Above average volume")
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error in pattern analysis for {symbol}: {e}")
        return {"error": f"Failed to analyze patterns: {str(e)}"}


# Securities Analysis Tools List
SECURITIES_ANALYSIS_TOOLS = [
    perform_technical_analysis,
    analyze_security_fundamentals,
    compare_securities,
    analyze_security_patterns,
]
