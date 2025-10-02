"""Production-ready hierarchical supervisor agent for investment advisory.

This module creates a multi-agent system with specialized agents for:
- Mathematical analysis and calculations
- Portfolio analysis and client data
- Market research and trend analysis

The system includes safety checks, tool integration, and extensible architecture.
"""
from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph_supervisor import create_supervisor

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.portfolio_tools import PORTFOLIO_TOOLS
from agents.tools import calculator
from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


# Initialize tools for production use
web_search = DuckDuckGoSearchResults(name="WebSearch")
base_tools = [web_search, calculator] + PORTFOLIO_TOOLS

# Add weather tool if API key is set
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    base_tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

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


# Math tools for specialized math agent
math_tools = [add, multiply, divide, subtract, calculator]

# Research tools for research agent
research_tools = [web_search] + PORTFOLIO_TOOLS
if settings.OPENWEATHERMAP_API_KEY:
    weather_wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    research_tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=weather_wrapper))


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    """Format safety violation message."""
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


def wrap_model_with_instructions(
    llm_model: BaseChatModel, instructions: str, tools: list
) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap model with system instructions and specific tools."""
    bound_model = llm_model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def safety_check_and_model(
    state: AgentState, config: RunnableConfig, instructions: str, tools: list
) -> AgentState:
    """Run safety check and model inference."""
    configurable = config.get("configurable", {})
    model_name = configurable.get("model", settings.DEFAULT_MODEL)
    llm_model = get_model(model_name)
    model_runnable = wrap_model_with_instructions(llm_model, instructions, tools)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    remaining_steps = state.get("remaining_steps", 10)
    if remaining_steps < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """Check input safety."""
    del config  # Unused parameter
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    """Block unsafe content."""
    del config  # Unused parameter
    safety: LlamaGuardOutput = state.get(
        "safety", LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
    )
    return {"messages": [format_safety_message(safety)]}


def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    """Check if content is safe."""
    safety: LlamaGuardOutput = state.get(
        "safety", LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
    )
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    """Check if there are pending tool calls."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


def create_production_agent(agent_name: str, tools: list, instructions: str) -> StateGraph:
    """Create a production-ready agent with safety checks and tool support."""
    del agent_name  # Unused parameter - name is set during compilation

    async def agent_model(state: AgentState, config: RunnableConfig) -> AgentState:
        return await safety_check_and_model(state, config, instructions, tools)

    # Create agent graph
    agent = StateGraph(AgentState)
    agent.add_node("model", agent_model)
    agent.add_node("tools", ToolNode(tools))
    agent.add_node("guard_input", llama_guard_input)
    agent.add_node("block_unsafe_content", block_unsafe_content)
    agent.set_entry_point("guard_input")

    # Add safety routing
    agent.add_conditional_edges(
        "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
    )

    # Always END after blocking unsafe content
    agent.add_edge("block_unsafe_content", END)

    # Always run "model" after "tools"
    agent.add_edge("tools", "model")

    # After "model", if there are tool calls, run "tools". Otherwise END.
    agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

    return agent


def workflow(chosen_model):
    """Create a production-ready hierarchical supervisor for investment advisory."""

    # Define specialized agent instructions
    math_instructions = f"""
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

    portfolio_instructions = f"""
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

    research_instructions = f"""
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
    
    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.
    
    Always include markdown-formatted links to citations. Only use links returned by tools.
    Delegate complex mathematical calculations to the math expert.
    Delegate portfolio-specific analysis to the portfolio expert.
    """

    # Create specialized agents using create_react_agent for consistency with supervisor
    # These agents will have consistent message flow with the supervisor pattern
    math_agent = create_react_agent(
        model=chosen_model,
        tools=math_tools,
        name="sub-agent-math_expert",
        prompt=math_instructions,
    ).with_config(tags=["skip_stream"])

    portfolio_agent = create_react_agent(
        model=chosen_model,
        tools=PORTFOLIO_TOOLS,
        name="sub-agent-portfolio_expert",
        prompt=portfolio_instructions,
    ).with_config(tags=["skip_stream"])

    research_agent = create_react_agent(
        model=chosen_model,
        tools=research_tools,
        name="sub-agent-research_expert",
        prompt=research_instructions,
    ).with_config(tags=["skip_stream"])

    # Create intermediate supervisor for specialized analysis
    analysis_supervisor = create_supervisor(
        [math_agent, portfolio_agent],
        model=chosen_model,
        prompt=(
            "You are an analysis supervisor managing mathematical and portfolio experts. "
            "For mathematical calculations, risk analysis, or financial computations, "
            "use math_expert. For portfolio analysis, client data, or investment performance, "
            "use portfolio_expert. Coordinate between them when both mathematical and "
            "portfolio analysis are needed."
        ),
        supervisor_name="supervisor-analysis_team",
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(name="sub-supervisor-analysis_team").with_config(tags=["skip_stream"])

    # Create main supervisor managing research and analysis teams
    main_supervisor = create_supervisor(
        [research_agent, analysis_supervisor],
        model=chosen_model,
        prompt=(
            f"You are the main investment advisory supervisor managing research and "
            f"analysis teams. Today's date is {current_date}. "
            f"Team capabilities: "
            f"- research_expert: Market research, news analysis, company research, "
            f"economic indicators "
            f"- analysis_team (supervisor): Mathematical calculations and portfolio analysis "
            f"  - math_expert: Financial calculations, risk metrics, statistical analysis "
            f"  - portfolio_expert: Client portfolio analysis, performance tracking, "
            f"asset allocation "
            f"Routing guidelines: "
            f"- For market research, news, or general investment information → research_expert "
            f"- For client portfolio analysis or mathematical calculations → analysis_team "
            f"- For complex queries requiring both research and analysis → coordinate "
            f"between teams "
            f"Always ensure comprehensive analysis by leveraging the appropriate specialists. "
            f"The goal is to provide actionable investment advice and insights."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    )

    return main_supervisor


langgraph_supervisor_hierarchy_agent = workflow(model).compile()
