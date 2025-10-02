from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.lazy_agent import LazyLoadingAgent
from agents.market_research_agent import market_research_agent
from agents.portfolio_agent import portfolio_agent
from agents.risk_optimization_agent import risk_optimization_agent
from agents.supervisor_agent import (
    supervisor_agent,
)
from schema import AgentInfo

DEFAULT_AGENT = "supervisor-agent"

# Type alias to handle LangGraph's different agent patterns
# - @entrypoint functions return Pregel
# - StateGraph().compile() returns CompiledStateGraph
AgentGraph = CompiledStateGraph | Pregel  # What get_agent() returns (always loaded)
AgentGraphLike = CompiledStateGraph | Pregel | LazyLoadingAgent  # What can be stored in registry


@dataclass
class Agent:
    description: str
    graph_like: AgentGraphLike


agents: dict[str, Agent] = {
    "portfolio-agent": Agent(
        description="An advanced Securities & Portfolio Analysis specialist combining deep technical analysis (RSI, MACD, Bollinger Bands) with comprehensive portfolio management, fundamental analysis, and quantitative optimization.",
        graph_like=portfolio_agent,
    ),
    "market-research-agent": Agent(
        description="A market research expert for trend analysis, company research, and economic indicators.",
        graph_like=market_research_agent,
    ),
    "risk-optimization-agent": Agent(
        description="A risk assessment and portfolio optimization specialist for risk analysis, compliance monitoring, and portfolio optimization.",
        graph_like=risk_optimization_agent,
    ),
    "supervisor-agent": Agent(
        description="A production-ready hierarchical supervisor managing specialized investment advisory agents",
        graph_like=supervisor_agent,
    ),
}


async def load_agent(agent_id: str) -> None:
    """Load lazy agents if needed."""
    graph_like = agents[agent_id].graph_like
    if isinstance(graph_like, LazyLoadingAgent):
        await graph_like.load()


def get_agent(agent_id: str) -> AgentGraph:
    """Get an agent graph, loading lazy agents if needed."""
    agent_graph = agents[agent_id].graph_like

    # If it's a lazy loading agent, ensure it's loaded and return its graph
    if isinstance(agent_graph, LazyLoadingAgent):
        if not agent_graph._loaded:
            raise RuntimeError(f"Agent {agent_id} not loaded. Call load() first.")
        return agent_graph.get_graph()

    # Otherwise return the graph directly
    return agent_graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
