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
from .portfolio_tools import PORTFOLIO_TOOLS
from .tools import calculator

current_date = datetime.now().strftime("%B %d, %Y")

# Initialize research tools
web_search = DuckDuckGoSearchResults(name="WebSearch")
research_tools = [web_search, calculator] + PORTFOLIO_TOOLS

# Add weather tool if API key is set
if settings.OPENWEATHERMAP_API_KEY:
    weather_wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    research_tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=weather_wrapper))

# Instructions for the market research agent
MARKET_RESEARCH_INSTRUCTIONS = f"""
You are a market research and analysis expert for investment advisory.
Today's date is {current_date}.

Your expertise includes:
- Market research and trend analysis
- Company and sector analysis
- Economic indicators and news analysis
- Investment opportunity identification
- Risk factor assessment

Tools available:
- Web search for current market information
- Weather data for sector-specific analysis (agriculture, energy, etc.)
- Calculator for basic computations
- Portfolio tools for client data analysis

NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

Always include markdown-formatted links to citations. Only use links returned by tools.
Delegate complex mathematical calculations to the math expert when working in a team.
Delegate portfolio-specific analysis to the portfolio expert when working in a team.
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


# Create the standalone market research agent
market_research_agent = create_market_research_agent()
market_research_agent = create_market_research_agent()
