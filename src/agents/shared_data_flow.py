"""Enhanced data flow system for inter-agent communication and coordination."""

import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AgentDataPacket(BaseModel):
    """Structured data packet for inter-agent communication."""
    
    packet_id: str
    source_agent: str
    target_agent: Optional[str] = None  # None means broadcast to all
    data_type: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(
        cls,
        source_agent: str,
        data_type: str,
        data: Dict[str, Any],
        target_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AgentDataPacket":
        """Create a new data packet."""
        return cls(
            packet_id=str(uuid.uuid4()),
            source_agent=source_agent,
            target_agent=target_agent,
            data_type=data_type,
            data=data,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )


class SharedDataCache:
    """Thread-safe shared data cache for agents."""
    
    def __init__(self):
        self._cache: Dict[str, List[AgentDataPacket]] = {}
        self._lock = threading.Lock()
        self._subscriptions: Dict[str, List[str]] = {}  # agent -> data_types
    
    def store_data(self, packet: AgentDataPacket, thread_id: str) -> None:
        """Store data packet in thread-specific cache."""
        with self._lock:
            if thread_id not in self._cache:
                self._cache[thread_id] = []
            self._cache[thread_id].append(packet)
    
    def get_data(
        self,
        thread_id: str,
        data_type: Optional[str] = None,
        source_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
        limit: int = 10
    ) -> List[AgentDataPacket]:
        """Retrieve data packets with optional filtering."""
        with self._lock:
            if thread_id not in self._cache:
                return []
            
            packets = self._cache[thread_id]
            
            # Apply filters
            if data_type:
                packets = [p for p in packets if p.data_type == data_type]
            if source_agent:
                packets = [p for p in packets if p.source_agent == source_agent]
            if target_agent:
                packets = [p for p in packets if p.target_agent == target_agent or p.target_agent is None]
            
            # Sort by timestamp (newest first) and limit
            packets = sorted(packets, key=lambda p: p.timestamp, reverse=True)
            return packets[:limit]
    
    def get_latest_data(self, thread_id: str, data_type: str) -> Optional[AgentDataPacket]:
        """Get the most recent data packet of a specific type."""
        packets = self.get_data(thread_id, data_type=data_type, limit=1)
        return packets[0] if packets else None
    
    def subscribe_agent(self, agent_name: str, data_types: List[str]) -> None:
        """Subscribe an agent to specific data types."""
        with self._lock:
            self._subscriptions[agent_name] = data_types
    
    def get_subscribed_data(self, thread_id: str, agent_name: str, limit: int = 20) -> List[AgentDataPacket]:
        """Get all data relevant to a subscribed agent."""
        with self._lock:
            if agent_name not in self._subscriptions:
                return []
            
            subscribed_types = self._subscriptions[agent_name]
            all_packets = []
            
            for data_type in subscribed_types:
                packets = self.get_data(thread_id, data_type=data_type, limit=limit)
                all_packets.extend(packets)
            
            # Remove duplicates and sort by timestamp
            seen_ids = set()
            unique_packets = []
            for packet in all_packets:
                if packet.packet_id not in seen_ids:
                    seen_ids.add(packet.packet_id)
                    unique_packets.append(packet)
            
            return sorted(unique_packets, key=lambda p: p.timestamp, reverse=True)
    
    def clear_thread_data(self, thread_id: str) -> None:
        """Clear all data for a specific thread."""
        with self._lock:
            if thread_id in self._cache:
                del self._cache[thread_id]


class DataFlowCoordinator:
    """Coordinates complex multi-agent workflows."""
    
    def __init__(self):
        self.workflows: Dict[str, List[Dict[str, Any]]] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def register_workflow(self, workflow_name: str, steps: List[Dict[str, Any]]) -> None:
        """Register a new workflow template."""
        with self._lock:
            self.workflows[workflow_name] = steps
    
    def start_workflow(self, thread_id: str, workflow_name: str, input_data: Dict[str, Any]) -> str:
        """Start a workflow execution."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow_id = str(uuid.uuid4())
        
        with self._lock:
            self.active_workflows[workflow_id] = {
                "thread_id": thread_id,
                "workflow_name": workflow_name,
                "steps": self.workflows[workflow_name].copy(),
                "current_step": 0,
                "status": "running",
                "input_data": input_data,
                "results": {},
                "started_at": datetime.now()
            }
        
        return workflow_id
    
    def execute_workflow_step(
        self,
        thread_id: str,
        workflow_name: str,
        step_index: int,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        if workflow_name not in self.workflows:
            return {"error": f"Unknown workflow: {workflow_name}"}
        
        steps = self.workflows[workflow_name]
        if step_index >= len(steps):
            return {"error": "Step index out of range"}
        
        step = steps[step_index]
        
        # Create data packet for this workflow step
        packet = AgentDataPacket.create(
            source_agent="workflow_coordinator",
            data_type="workflow_step",
            data={
                "workflow_name": workflow_name,
                "step_index": step_index,
                "step_config": step,
                "input_data": input_data
            },
            metadata={"thread_id": thread_id}
        )
        
        return {
            "step_executed": True,
            "step_config": step,
            "packet_id": packet.packet_id
        }


# Global instances
shared_cache = SharedDataCache()
workflow_coordinator = DataFlowCoordinator()


# Pre-defined workflows for common investment advisory tasks
INVESTMENT_ADVISORY_WORKFLOWS = {
    "comprehensive_portfolio_analysis": [
        {"agent": "portfolio_expert", "task": "analyze_portfolio_holdings"},
        {"agent": "risk_optimization_expert", "task": "assess_portfolio_risk"},
        {"agent": "market_research_expert", "task": "research_portfolio_securities"},
        {"agent": "math_expert", "task": "calculate_risk_metrics"},
        {"agent": "risk_optimization_expert", "task": "generate_optimization_recommendations"}
    ],
    "market_condition_assessment": [
        {"agent": "market_research_expert", "task": "analyze_current_market"},
        {"agent": "risk_optimization_expert", "task": "assess_economic_risks"},
        {"agent": "math_expert", "task": "calculate_market_metrics"},
        {"agent": "portfolio_expert", "task": "evaluate_portfolio_impact"}
    ],
    "client_risk_evaluation": [
        {"agent": "portfolio_expert", "task": "get_client_profile"},
        {"agent": "risk_optimization_expert", "task": "evaluate_risk_tolerance"},
        {"agent": "market_research_expert", "task": "analyze_market_conditions"},
        {"agent": "risk_optimization_expert", "task": "generate_risk_recommendations"}
    ]
}

# Register pre-defined workflows
for workflow_name, steps in INVESTMENT_ADVISORY_WORKFLOWS.items():
    workflow_coordinator.register_workflow(workflow_name, steps)


def get_shared_data_summary(thread_id: str) -> Dict[str, Any]:
    """Get a summary of all shared data in the current thread."""
    all_data = shared_cache.get_data(thread_id, limit=100)
    
    # Group by data type
    data_by_type = {}
    for packet in all_data:
        data_type = packet.data_type
        if data_type not in data_by_type:
            data_by_type[data_type] = []
        data_by_type[data_type].append({
            "packet_id": packet.packet_id,
            "source_agent": packet.source_agent,
            "timestamp": packet.timestamp.isoformat(),
            "data_keys": list(packet.data.keys())
        })
    
    # Group by source agent
    data_by_agent = {}
    for packet in all_data:
        agent = packet.source_agent
        if agent not in data_by_agent:
            data_by_agent[agent] = []
        data_by_agent[agent].append({
            "packet_id": packet.packet_id,
            "data_type": packet.data_type,
            "timestamp": packet.timestamp.isoformat()
        })
    
    summary = {
        "thread_id": thread_id,
        "total_packets": len(all_data),
        "data_types": list(data_by_type.keys()),
        "source_agents": list(data_by_agent.keys()),
        "data_by_type": data_by_type,
        "data_by_agent": data_by_agent,
        "latest_activity": all_data[0].timestamp.isoformat() if all_data else None
    }
    
    return summary


# Agent subscriptions - define what data types each agent is interested in
AGENT_SUBSCRIPTIONS = {
    "portfolio_expert": [
        "client_data", "portfolio_analysis", "market_research", "risk_assessment"
    ],
    "market_research_expert": [
        "portfolio_holdings", "client_securities", "economic_data", "market_analysis"
    ],
    "risk_optimization_expert": [
        "portfolio_analysis", "market_research", "client_profile", "economic_data"
    ],
    "math_expert": [
        "portfolio_data", "market_data", "risk_data", "calculation_requests"
    ]
}

# Register agent subscriptions
for agent_name, data_types in AGENT_SUBSCRIPTIONS.items():
    shared_cache.subscribe_agent(agent_name, data_types)
