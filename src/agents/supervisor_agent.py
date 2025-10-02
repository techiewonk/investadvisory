"""Production-ready hierarchical supervisor agent for investment advisory.

This module creates a multi-agent system with specialized agents for:
- Mathematical analysis and calculations
- Portfolio analysis and client data
- Market research and trend analysis

The system includes safety checks, tool integration, and extensible architecture.
"""
from datetime import datetime
from typing import Literal

# Removed - now imported from individual agent modules
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

from core import get_model, settings

from .llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from .market_research_agent import create_market_research_agent
from .math_agent import MATH_TOOLS, create_math_agent
from .portfolio_agent import create_portfolio_agent
from .risk_optimization_agent import create_risk_optimization_agent

model = get_model(settings.DEFAULT_MODEL)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


current_date = datetime.now().strftime("%B %d, %Y")


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

    # Create specialized agents using the imported agent creation functions
    # These agents are now defined in separate files for independent use
    math_agent = create_math_agent(chosen_model).with_config(tags=["skip_stream"])
    portfolio_agent_instance = create_portfolio_agent(chosen_model).with_config(tags=["skip_stream"])
    research_agent = create_market_research_agent(chosen_model).with_config(tags=["skip_stream"])
    risk_optimization_agent_instance = create_risk_optimization_agent(chosen_model).with_config(tags=["skip_stream"])

    # Create intermediate supervisor for specialized analysis
    analysis_supervisor = create_supervisor(
        [math_agent, portfolio_agent_instance],
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
        [research_agent, analysis_supervisor, risk_optimization_agent_instance],
        model=chosen_model,
        prompt=(
            f"You are the main investment advisory supervisor managing research, "
            f"analysis, and risk optimization teams. Today's date is {current_date}. "
            f"Team capabilities: "
            f"- research_expert: Market research, news analysis, company research, "
            f"economic indicators "
            f"- analysis_team (supervisor): Mathematical calculations and portfolio analysis "
            f"  - math_expert: Financial calculations, risk metrics, statistical analysis "
            f"  - portfolio_expert: Client portfolio analysis, performance tracking, "
            f"asset allocation "
            f"- risk_optimization_expert: Risk assessment, compliance monitoring, "
            f"portfolio optimization, stress testing, regulatory compliance "
            f"Routing guidelines: "
            f"- For market research, news, or general investment information → research_expert "
            f"- For client portfolio analysis or mathematical calculations → analysis_team "
            f"- For risk assessment, compliance checks, or portfolio optimization → risk_optimization_expert "
            f"- For complex queries requiring multiple teams → coordinate between teams "
            f"Always ensure comprehensive analysis by leveraging the appropriate specialists. "
            f"The goal is to provide actionable investment advice and insights."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    )

    return main_supervisor


supervisor_agent = workflow(model).compile()

