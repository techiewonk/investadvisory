"""Data flow tools for inter-agent communication and coordination."""

from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .shared_data_flow import (
    AGENT_SUBSCRIPTIONS,
    INVESTMENT_ADVISORY_WORKFLOWS,
    AgentDataPacket,
    shared_cache,
    workflow_coordinator,
)


def get_thread_id_from_config(config: RunnableConfig) -> str:
    """Extract thread_id from RunnableConfig."""
    if not config:
        return "default"
    configurable = config.get("configurable", {})
    return configurable.get('thread_id', 'default')


@tool
def share_data_with_agents(
    data_type: str,
    data: Dict[str, Any],
    source_agent: str,
    target_agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: RunnableConfig = None
) -> Dict[str, Any]:
    """
    Share data with other agents in the system.
    
    Args:
        data_type: Type of data being shared (e.g., 'portfolio_analysis', 'market_research')
        data: The actual data to share
        source_agent: Name of the agent sharing the data
        target_agent: Specific target agent (None for broadcast to all)
        metadata: Additional metadata about the data
        config: Runtime configuration (automatically injected)
        
    Returns:
        Dictionary confirming data was shared
    """
    try:
        thread_id = get_thread_id_from_config(config)
        
        # Create data packet
        packet = AgentDataPacket.create(
            source_agent=source_agent,
            data_type=data_type,
            data=data,
            target_agent=target_agent,
            metadata=metadata
        )
        
        # Store in shared cache
        shared_cache.store_data(packet, thread_id)
        
        return {
            "success": True,
            "packet_id": packet.packet_id,
            "data_type": data_type,
            "source_agent": source_agent,
            "target_agent": target_agent or "all_agents",
            "timestamp": packet.timestamp.isoformat(),
            "message": f"Data shared successfully with {target_agent or 'all agents'}"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def get_shared_data(
    data_type: Optional[str] = None,
    source_agent: Optional[str] = None,
    limit: int = 10,
    config: RunnableConfig = None
) -> Dict[str, Any]:
    """
    Retrieve shared data from other agents.
    
    Args:
        data_type: Filter by data type (optional)
        source_agent: Filter by source agent (optional)
        limit: Maximum number of data packets to retrieve
        config: Runtime configuration (automatically injected)
        
    Returns:
        Dictionary containing retrieved data packets
    """
    try:
        thread_id = get_thread_id_from_config(config)
        
        # Get data from shared cache
        packets = shared_cache.get_data(
            thread_id=thread_id,
            data_type=data_type,
            source_agent=source_agent,
            limit=limit
        )
        
        # Convert packets to serializable format
        data_packets = []
        for packet in packets:
            data_packets.append({
                "packet_id": packet.packet_id,
                "source_agent": packet.source_agent,
                "target_agent": packet.target_agent,
                "data_type": packet.data_type,
                "data": packet.data,
                "timestamp": packet.timestamp.isoformat(),
                "metadata": packet.metadata
            })
        
        return {
            "success": True,
            "thread_id": thread_id,
            "packets_found": len(data_packets),
            "data_packets": data_packets,
            "filters_applied": {
                "data_type": data_type,
                "source_agent": source_agent,
                "limit": limit
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def get_agent_subscriptions(agent_name: str, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Get all data relevant to a specific agent based on their subscriptions.
    
    Args:
        agent_name: Name of the agent to get subscriptions for
        config: Runtime configuration (automatically injected)
        
    Returns:
        Dictionary containing subscribed data for the agent
    """
    try:
        thread_id = get_thread_id_from_config(config)
        
        # Get subscribed data
        packets = shared_cache.get_subscribed_data(thread_id, agent_name)
        
        # Convert to serializable format
        subscribed_data = []
        for packet in packets:
            subscribed_data.append({
                "packet_id": packet.packet_id,
                "source_agent": packet.source_agent,
                "data_type": packet.data_type,
                "data": packet.data,
                "timestamp": packet.timestamp.isoformat(),
                "metadata": packet.metadata
            })
        
        # Get subscription info
        subscriptions = AGENT_SUBSCRIPTIONS.get(agent_name, [])
        
        return {
            "success": True,
            "agent_name": agent_name,
            "subscriptions": subscriptions,
            "data_packets_found": len(subscribed_data),
            "subscribed_data": subscribed_data,
            "thread_id": thread_id
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def coordinate_workflow(
    workflow_name: str,
    input_data: Dict[str, Any],
    config: RunnableConfig = None
) -> Dict[str, Any]:
    """
    Initiate a coordinated multi-agent workflow.
    
    Args:
        workflow_name: Name of the workflow to execute
        input_data: Input data for the workflow
        config: Runtime configuration (automatically injected)
        
    Returns:
        Dictionary containing workflow coordination details
    """
    try:
        thread_id = get_thread_id_from_config(config)
        
        # Check if workflow exists
        if workflow_name not in INVESTMENT_ADVISORY_WORKFLOWS:
            available_workflows = list(INVESTMENT_ADVISORY_WORKFLOWS.keys())
            return {
                "success": False,
                "error": f"Unknown workflow: {workflow_name}",
                "available_workflows": available_workflows
            }
        
        # Start workflow
        workflow_id = workflow_coordinator.start_workflow(
            thread_id=thread_id,
            workflow_name=workflow_name,
            input_data=input_data
        )
        
        # Get workflow steps
        steps = INVESTMENT_ADVISORY_WORKFLOWS[workflow_name]
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "thread_id": thread_id,
            "total_steps": len(steps),
            "workflow_steps": steps,
            "status": "initiated",
            "message": f"Workflow '{workflow_name}' initiated with {len(steps)} steps"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def get_data_flow_summary(config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Get a summary of all data flow activity in the current session.
    
    Args:
        config: Runtime configuration (automatically injected)
        
    Returns:
        Dictionary containing data flow summary
    """
    try:
        thread_id = get_thread_id_from_config(config)
        
        # Get comprehensive summary
        from .shared_data_flow import get_shared_data_summary
        summary = get_shared_data_summary(thread_id)
        
        # Add workflow information
        available_workflows = list(INVESTMENT_ADVISORY_WORKFLOWS.keys())
        agent_subscriptions = AGENT_SUBSCRIPTIONS
        
        enhanced_summary = {
            **summary,
            "available_workflows": available_workflows,
            "agent_subscriptions": agent_subscriptions,
            "workflow_descriptions": {
                "comprehensive_portfolio_analysis": "5-step analysis: portfolio → risk → research → math → optimization",
                "market_condition_assessment": "4-step assessment: market → economic risks → metrics → portfolio impact",
                "client_risk_evaluation": "4-step evaluation: profile → tolerance → market → recommendations"
            }
        }
        
        return {
            "success": True,
            "data_flow_summary": enhanced_summary
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# Data flow tools list
DATA_FLOW_TOOLS = [
    share_data_with_agents,
    get_shared_data,
    get_agent_subscriptions,
    coordinate_workflow,
    get_data_flow_summary,
]
