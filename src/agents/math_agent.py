"""Mathematical analysis agent for investment advisory.

This agent specializes in:
- Portfolio calculations and risk analysis
- Financial mathematics and compound interest
- Statistical analysis and correlation calculations
- Options pricing and derivatives math
- Performance metrics and ratios
"""
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, create_react_agent

from core import get_model, settings

from .llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from .tools import calculator

current_date = datetime.now().strftime("%B %d, %Y")


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        return float('inf')
    return a / b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


# Math tools for the math agent
MATH_TOOLS = [add, multiply, divide, subtract, calculator]

# Instructions for the math agent
MATH_INSTRUCTIONS = f"""
You are a specialized mathematical analysis expert for investment advisory.
Today's date is {current_date}.

Your expertise includes:
- Portfolio calculations and risk analysis
- Financial mathematics and compound interest
- Statistical analysis and correlation calculations
- Options pricing and derivatives math
- Performance metrics and ratios

Always use appropriate tools for calculations. Be precise and show your work.
Focus only on mathematical aspects - delegate research and data gathering to other agents.
"""


def create_math_agent(model=None):
    """Create a mathematical analysis agent using create_react_agent."""
    if model is None:
        model = get_model(settings.DEFAULT_MODEL)
    
    return create_react_agent(
        model=model,
        tools=MATH_TOOLS,
        name="math_agent",
        prompt=MATH_INSTRUCTIONS,
    )


# Create the standalone math agent
math_agent = create_math_agent()
math_agent = create_math_agent()
