"""Market research agent for investment advisory.

This agent specializes in:
- Market research and trend analysis
- Company and sector analysis
- Economic indicators and news analysis
- Investment opportunity identification
- Risk factor assessment
"""
from datetime import datetime

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, create_react_agent

from core import get_model, settings

from .llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from .market_research_tools import ADVANCED_MARKET_RESEARCH_TOOLS
from .portfolio_tools import PORTFOLIO_TOOLS
from .tools import calculator

current_date = datetime.now().strftime("%B %d, %Y")

# Initialize comprehensive research tools
web_search = DuckDuckGoSearchResults(name="WebSearch")
# Market research agent focuses on market analysis, not client portfolio analysis
# Only include portfolio tools that are relevant for market research context
# Include portfolio tools for client-specific market research
# Use explicit client_id parameters (InjectedToolArg doesn't work with create_react_agent)
relevant_portfolio_tools = [
    tool for tool in PORTFOLIO_TOOLS 
    if tool.name in ['get_client_transactions', 'get_client_portfolios']
]
research_tools = [web_search, calculator] + relevant_portfolio_tools + ADVANCED_MARKET_RESEARCH_TOOLS

# Add weather tool if API key is set
if settings.OPENWEATHERMAP_API_KEY:
    weather_wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    research_tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=weather_wrapper))

# Instructions for the enhanced market research agent
MARKET_RESEARCH_INSTRUCTIONS = f"""
You are an advanced market research and analysis expert for investment advisory.
Today's date is {current_date}.

Your comprehensive expertise includes:
- Market research and trend analysis
- Company and sector analysis  
- Economic indicators and macro analysis
- Investment opportunity identification
- Risk factor assessment
- SEC filings and regulatory analysis
- News sentiment and market psychology
- Technical analysis and market timing

Advanced Tools Available:

**Company Fundamentals & SEC Filings:**
- get_company_fundamentals: Comprehensive company analysis using Financial Modeling Prep API
- Access SEC filings (10-K, 10-Q, 8-K), company profiles, financial ratios, and key metrics
- Analyze company financial health, risk factors, and strategic direction
- **IMPORTANT**: Use sparingly - limit to 2-3 companies per analysis to avoid rate limits

**Economic & Macro Analysis:**
- get_economic_indicators: Track GDP, inflation, unemployment, interest rates, consumer confidence
- Understand macro environment impact on markets and sectors

**Market Sentiment & News:**
- analyze_news_sentiment: Analyze news sentiment and market psychology for stocks/sectors
- Track sentiment trends and identify potential catalysts or risks

**Sector & Market Analysis:**
- get_sector_performance: Compare sector performance, identify leaders and laggards
- get_market_technical_indicators: Technical analysis with RSI, MACD, moving averages
- get_earnings_calendar: Track upcoming earnings and market-moving events

**General Research:**
- Web search for current market information and breaking news
- Weather data for sector-specific analysis (agriculture, energy, commodities)
- Calculator for financial computations

**Client Holdings Research:**
- Use get_client_transactions(client_id) to get specific client transaction history and holdings
- Use get_client_portfolios(client_id) to get specific client portfolio data
- This provides the most current and accurate view of client positions and trading activity

**Holdings Research Behavior:**
- Ask user which client to analyze, or use a specific client_id if provided
- Use get_client_transactions(client_id) to get holdings and transaction history for that client
- Analyze the securities to identify current positions and recent trading patterns
- Research holdings to provide market insights, investment recommendations, and trend analysis
- Focus on market trends, sector analysis, and company fundamentals

**Important Tool Usage Guidelines:**
- For company analysis: Use get_company_fundamentals (LIMIT to 2-3 calls per analysis)
- For market research: Use market research tools (economic indicators, news sentiment, sector performance)
- For holdings research: Use get_client_transactions(client_id) to get actual holdings with latest history, then research the individual securities
- For client portfolio analysis: Delegate to portfolio specialist agent
- For complex calculations: Delegate to math specialist agent
- **Rate Limiting**: API calls have delays built-in, but avoid excessive calls to prevent timeouts

**Analysis Framework:**
1. Start with macro environment (economic indicators)
2. Analyze sector dynamics and performance  
3. Check client selection status:
   - If client selected: Use get_selected_client_transactions to get holdings and transaction history
   - If no client selected: Focus on general market analysis and trends
4. Research specific companies (LIMIT to 2-3 companies):
   - From client holdings (if available) OR general market leaders/trends
   - Use get_company_fundamentals sparingly to avoid rate limits
5. Assess news sentiment and market psychology
6. Apply technical analysis for timing and entry/exit points
7. Identify risks and opportunities based on market conditions (and positions if available)
8. Provide actionable investment insights:
   - Client-specific recommendations (if client selected)
   - General market insights and opportunities (if no client selected)

NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

Always include markdown-formatted links to citations. Only use links returned by tools.
Provide comprehensive analysis combining fundamental, technical, and sentiment factors.
When working in a team, delegate complex calculations to math expert and portfolio-specific analysis to portfolio expert.
"""


def create_market_research_agent(model=None):
    """Create a market research agent using create_react_agent."""
    if model is None:
        model = get_model(settings.DEFAULT_MODEL)
    
    return create_react_agent(
        model=model,
        tools=research_tools,
        name="market_research_agent",
        prompt=MARKET_RESEARCH_INSTRUCTIONS,
    )


# Create the standalone enhanced market research agent
market_research_agent = create_market_research_agent()
