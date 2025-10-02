"""Portfolio analysis agent for investment advisory.

This agent specializes in:
- Client portfolio analysis and performance tracking
- Asset allocation and diversification analysis
- Risk assessment and portfolio optimization
- Transaction history analysis
- Holdings performance evaluation
"""
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, create_react_agent

from core import get_model, settings

from .llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from .portfolio_tools import PORTFOLIO_TOOLS

current_date = datetime.now().strftime("%B %d, %Y")

# Instructions for the portfolio agent
PORTFOLIO_INSTRUCTIONS = f"""
You are a portfolio analysis specialist for investment advisory.
Today's date is {current_date}.

Your expertise includes:
- Client portfolio analysis and performance tracking
- Asset allocation and diversification analysis
- Risk assessment and portfolio optimization
- Transaction history analysis
- Holdings performance evaluation

Portfolio Analysis Tools:
- If a client is selected in the UI, use get_selected_client_portfolios, get_selected_client_transactions, 
  or analyze_selected_client_performance for automatic analysis
- For specific clients, use get_client_portfolios, get_client_transactions, or analyze_client_portfolio_performance
- Use get_all_clients to see available clients

Always provide actionable insights and recommendations based on portfolio data.
"""


def create_portfolio_agent(model=None):
    """Create a portfolio analysis agent using create_react_agent."""
    if model is None:
        model = get_model(settings.DEFAULT_MODEL)
    
    return create_react_agent(
        model=model,
        tools=PORTFOLIO_TOOLS,
        name="portfolio_agent",
        prompt=PORTFOLIO_INSTRUCTIONS,
    )


# Create the standalone portfolio agent
portfolio_agent = create_portfolio_agent()
portfolio_agent = create_portfolio_agent()
