"""Agent Evaluation System for Investment Advisory Platform.

This module provides comprehensive evaluation and monitoring of agent performance.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentMetrics(BaseModel):
    """Metrics for a single agent."""
    
    agent_id: str = Field(description="Agent identifier")
    total_invocations: int = Field(default=0, description="Total number of invocations")
    successful_invocations: int = Field(default=0, description="Number of successful invocations")
    failed_invocations: int = Field(default=0, description="Number of failed invocations")
    average_response_time: float = Field(default=0.0, description="Average response time in seconds")
    total_response_time: float = Field(default=0.0, description="Total response time in seconds")
    confidence_scores: List[float] = Field(default_factory=list, description="Confidence scores")
    tool_usage: Dict[str, int] = Field(default_factory=dict, description="Tool usage count")
    error_types: Dict[str, int] = Field(default_factory=dict, description="Error type counts")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class EvaluationResult(BaseModel):
    """Result of agent evaluation."""
    
    agent_id: str = Field(description="Agent identifier")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall performance score")
    performance_metrics: Dict[str, Any] = Field(description="Individual performance metrics")
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Response accuracy score")
    reliability_score: float = Field(ge=0.0, le=1.0, description="System reliability score")
    efficiency_score: float = Field(ge=0.0, le=1.0, description="Response efficiency score")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    strengths: List[str] = Field(default_factory=list, description="Agent strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")


class SystemReport(BaseModel):
    """System-wide evaluation report."""
    
    report_timestamp: datetime = Field(default_factory=datetime.now)
    total_agents: int = Field(description="Total number of agents")
    active_agents: int = Field(description="Number of active agents")
    system_uptime: float = Field(description="System uptime in hours")
    total_queries: int = Field(description="Total queries processed")
    successful_queries: int = Field(description="Successful queries")
    average_system_response_time: float = Field(description="Average system response time")
    agent_evaluations: List[EvaluationResult] = Field(description="Individual agent evaluations")
    system_health_score: float = Field(ge=0.0, le=1.0, description="Overall system health score")
    recommendations: List[str] = Field(default_factory=list, description="System-wide recommendations")


class AgentEvaluator:
    """Comprehensive agent evaluation and monitoring system."""
    
    def __init__(self):
        self.metrics_store = defaultdict(lambda: AgentMetrics(agent_id=""))
        self.evaluation_history = []
        self.system_start_time = datetime.now()
        self.performance_benchmarks = {
            'response_time': {
                'excellent': 2.0,
                'good': 5.0,
                'acceptable': 10.0
            },
            'success_rate': {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70
            },
            'confidence': {
                'excellent': 0.9,
                'good': 0.7,
                'acceptable': 0.5
            }
        }
    
    def record_invocation(
        self,
        agent_id: str,
        response_time: float,
        success: bool,
        confidence_score: Optional[float] = None,
        tools_used: Optional[List[str]] = None,
        error_type: Optional[str] = None
    ) -> None:
        """
        Record an agent invocation for evaluation.
        
        Args:
            agent_id: Agent identifier
            response_time: Response time in seconds
            success: Whether the invocation was successful
            confidence_score: Confidence score of the response
            tools_used: List of tools used in the invocation
            error_type: Type of error if invocation failed
        """
        metrics = self.metrics_store[agent_id]
        if not metrics.agent_id:
            metrics.agent_id = agent_id
        
        # Update invocation counts
        metrics.total_invocations += 1
        if success:
            metrics.successful_invocations += 1
        else:
            metrics.failed_invocations += 1
        
        # Update response time metrics
        metrics.total_response_time += response_time
        metrics.average_response_time = metrics.total_response_time / metrics.total_invocations
        
        # Update confidence scores
        if confidence_score is not None:
            metrics.confidence_scores.append(confidence_score)
            # Keep only last 100 scores to prevent memory bloat
            if len(metrics.confidence_scores) > 100:
                metrics.confidence_scores = metrics.confidence_scores[-100:]
        
        # Update tool usage
        if tools_used:
            for tool in tools_used:
                metrics.tool_usage[tool] = metrics.tool_usage.get(tool, 0) + 1
        
        # Update error types
        if error_type:
            metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
        
        metrics.last_updated = datetime.now()
    
    def evaluate_agent(self, agent_id: str) -> EvaluationResult:
        """
        Evaluate a single agent's performance.
        
        Args:
            agent_id: Agent to evaluate
            
        Returns:
            EvaluationResult with comprehensive evaluation
        """
        metrics = self.metrics_store[agent_id]
        
        if metrics.total_invocations == 0:
            return EvaluationResult(
                agent_id=agent_id,
                overall_score=0.0,
                performance_metrics={},
                accuracy_score=0.0,
                reliability_score=0.0,
                efficiency_score=0.0,
                recommendations=["No data available for evaluation"]
            )
        
        # Calculate individual scores
        accuracy_score = self._calculate_accuracy_score(metrics)
        reliability_score = self._calculate_reliability_score(metrics)
        efficiency_score = self._calculate_efficiency_score(metrics)
        
        # Calculate overall score (weighted average)
        overall_score = (
            accuracy_score * 0.4 +
            reliability_score * 0.4 +
            efficiency_score * 0.2
        )
        
        # Generate performance metrics
        performance_metrics = {
            'success_rate': metrics.successful_invocations / metrics.total_invocations,
            'average_response_time': metrics.average_response_time,
            'average_confidence': np.mean(metrics.confidence_scores) if metrics.confidence_scores else 0.0,
            'total_invocations': float(metrics.total_invocations),
            'most_used_tool': max(metrics.tool_usage.items(), key=lambda x: x[1])[0] if metrics.tool_usage else "none",
            'error_rate': metrics.failed_invocations / metrics.total_invocations
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, accuracy_score, reliability_score, efficiency_score)
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(metrics, performance_metrics)
        
        result = EvaluationResult(
            agent_id=agent_id,
            overall_score=overall_score,
            performance_metrics=performance_metrics,
            accuracy_score=accuracy_score,
            reliability_score=reliability_score,
            efficiency_score=efficiency_score,
            recommendations=recommendations,
            strengths=strengths,
            weaknesses=weaknesses
        )
        
        self.evaluation_history.append(result)
        return result
    
    def generate_system_report(self) -> SystemReport:
        """
        Generate a comprehensive system evaluation report.
        
        Returns:
            SystemReport with system-wide metrics and evaluations
        """
        # Get all agent evaluations
        agent_evaluations = []
        total_queries = 0
        successful_queries = 0
        total_response_time = 0.0
        
        for agent_id in self.metrics_store.keys():
            evaluation = self.evaluate_agent(agent_id)
            agent_evaluations.append(evaluation)
            
            metrics = self.metrics_store[agent_id]
            total_queries += metrics.total_invocations
            successful_queries += metrics.successful_invocations
            total_response_time += metrics.total_response_time
        
        # Calculate system metrics
        active_agents = len([m for m in self.metrics_store.values() if m.total_invocations > 0])
        system_uptime = (datetime.now() - self.system_start_time).total_seconds() / 3600  # hours
        average_system_response_time = total_response_time / total_queries if total_queries > 0 else 0.0
        
        # Calculate system health score
        if agent_evaluations:
            system_health_score = np.mean([eval.overall_score for eval in agent_evaluations])
        else:
            system_health_score = 0.0
        
        # Generate system-wide recommendations
        system_recommendations = self._generate_system_recommendations(agent_evaluations)
        
        return SystemReport(
            total_agents=len(self.metrics_store),
            active_agents=active_agents,
            system_uptime=system_uptime,
            total_queries=total_queries,
            successful_queries=successful_queries,
            average_system_response_time=average_system_response_time,
            agent_evaluations=agent_evaluations,
            system_health_score=system_health_score,
            recommendations=system_recommendations
        )
    
    def _calculate_accuracy_score(self, metrics: AgentMetrics) -> float:
        """Calculate accuracy score based on confidence and success rate."""
        success_rate = metrics.successful_invocations / metrics.total_invocations
        avg_confidence = np.mean(metrics.confidence_scores) if metrics.confidence_scores else 0.5
        
        # Weighted combination of success rate and confidence
        accuracy_score = success_rate * 0.7 + avg_confidence * 0.3
        return min(1.0, accuracy_score)
    
    def _calculate_reliability_score(self, metrics: AgentMetrics) -> float:
        """Calculate reliability score based on consistency and error rates."""
        success_rate = metrics.successful_invocations / metrics.total_invocations
        
        # Penalty for high error rates
        error_penalty = 0.0
        if metrics.error_types:
            total_errors = sum(metrics.error_types.values())
            error_penalty = min(0.3, total_errors / metrics.total_invocations * 0.5)
        
        # Consistency bonus for stable confidence scores
        consistency_bonus = 0.0
        if len(metrics.confidence_scores) > 5:
            confidence_std = np.std(metrics.confidence_scores)
            consistency_bonus = max(0.0, 0.1 - confidence_std)
        
        reliability_score = success_rate - error_penalty + consistency_bonus
        return max(0.0, min(1.0, reliability_score))
    
    def _calculate_efficiency_score(self, metrics: AgentMetrics) -> float:
        """Calculate efficiency score based on response time."""
        avg_response_time = metrics.average_response_time
        benchmarks = self.performance_benchmarks['response_time']
        
        if avg_response_time <= benchmarks['excellent']:
            return 1.0
        elif avg_response_time <= benchmarks['good']:
            return 0.8
        elif avg_response_time <= benchmarks['acceptable']:
            return 0.6
        else:
            # Linear decay after acceptable threshold
            return max(0.0, 0.6 - (avg_response_time - benchmarks['acceptable']) * 0.05)
    
    def _generate_recommendations(
        self,
        metrics: AgentMetrics,
        accuracy_score: float,
        reliability_score: float,
        efficiency_score: float
    ) -> List[str]:
        """Generate improvement recommendations for an agent."""
        recommendations = []
        
        # Accuracy recommendations
        if accuracy_score < 0.7:
            avg_confidence = np.mean(metrics.confidence_scores) if metrics.confidence_scores else 0.0
            if avg_confidence < 0.6:
                recommendations.append("Improve response confidence through better data validation")
            success_rate = metrics.successful_invocations / metrics.total_invocations
            if success_rate < 0.8:
                recommendations.append("Investigate and fix common failure patterns")
        
        # Reliability recommendations
        if reliability_score < 0.7:
            if metrics.error_types:
                most_common_error = max(metrics.error_types.items(), key=lambda x: x[1])
                recommendations.append(f"Address frequent {most_common_error[0]} errors")
            
            if len(metrics.confidence_scores) > 5:
                confidence_std = np.std(metrics.confidence_scores)
                if confidence_std > 0.2:
                    recommendations.append("Improve response consistency")
        
        # Efficiency recommendations
        if efficiency_score < 0.7:
            if metrics.average_response_time > self.performance_benchmarks['response_time']['acceptable']:
                recommendations.append("Optimize response time through caching or algorithm improvements")
                
                # Tool-specific recommendations
                if metrics.tool_usage:
                    most_used_tool = max(metrics.tool_usage.items(), key=lambda x: x[1])[0]
                    recommendations.append(f"Optimize {most_used_tool} tool performance")
        
        # General recommendations
        if metrics.total_invocations < 10:
            recommendations.append("Insufficient data for comprehensive evaluation - continue monitoring")
        
        return recommendations
    
    def _identify_strengths_weaknesses(
        self,
        metrics: AgentMetrics,
        performance_metrics: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """Identify agent strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        # Success rate analysis
        success_rate = performance_metrics['success_rate']
        if success_rate >= 0.9:
            strengths.append("High success rate")
        elif success_rate < 0.7:
            weaknesses.append("Low success rate")
        
        # Response time analysis
        avg_response_time = performance_metrics['average_response_time']
        if avg_response_time <= self.performance_benchmarks['response_time']['excellent']:
            strengths.append("Excellent response time")
        elif avg_response_time > self.performance_benchmarks['response_time']['acceptable']:
            weaknesses.append("Slow response time")
        
        # Confidence analysis
        avg_confidence = performance_metrics['average_confidence']
        if avg_confidence >= 0.8:
            strengths.append("High confidence responses")
        elif avg_confidence < 0.6:
            weaknesses.append("Low confidence responses")
        
        # Tool usage analysis
        if metrics.tool_usage:
            tool_count = len(metrics.tool_usage)
            if tool_count >= 5:
                strengths.append("Diverse tool utilization")
            elif tool_count <= 2:
                weaknesses.append("Limited tool usage")
        
        return strengths, weaknesses
    
    def _generate_system_recommendations(self, agent_evaluations: List[EvaluationResult]) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        if not agent_evaluations:
            return ["No agent data available for system evaluation"]
        
        # Overall performance analysis
        avg_system_score = np.mean([eval.overall_score for eval in agent_evaluations])
        if avg_system_score < 0.7:
            recommendations.append("System performance below acceptable threshold - review agent configurations")
        
        # Identify underperforming agents
        poor_performers = [eval for eval in agent_evaluations if eval.overall_score < 0.6]
        if poor_performers:
            agent_list = ", ".join([eval.agent_id for eval in poor_performers])
            recommendations.append(f"Focus improvement efforts on underperforming agents: {agent_list}")
        
        # Response time analysis
        avg_response_times = [eval.performance_metrics.get('average_response_time', 0) for eval in agent_evaluations]
        if avg_response_times and np.mean(avg_response_times) > 10.0:
            recommendations.append("System-wide response time optimization needed")
        
        # Success rate analysis
        success_rates = [eval.performance_metrics.get('success_rate', 0) for eval in agent_evaluations]
        if success_rates and np.mean(success_rates) < 0.8:
            recommendations.append("Investigate system-wide reliability issues")
        
        return recommendations
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent."""
        return self.metrics_store.get(agent_id)
    
    def reset_metrics(self, agent_id: Optional[str] = None) -> None:
        """Reset metrics for an agent or all agents."""
        if agent_id:
            if agent_id in self.metrics_store:
                del self.metrics_store[agent_id]
        else:
            self.metrics_store.clear()
            self.evaluation_history.clear()
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        if format.lower() == 'json':
            export_data = {
                'system_start_time': self.system_start_time.isoformat(),
                'metrics': {
                    agent_id: metrics.dict() 
                    for agent_id, metrics in self.metrics_store.items()
                },
                'evaluation_history': [eval.dict() for eval in self.evaluation_history[-10:]]  # Last 10
            }
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global evaluator instance
agent_evaluator = AgentEvaluator()


def track_agent_performance(agent_id: str):
    """
    Decorator to track agent performance metrics.
    
    Args:
        agent_id: Agent identifier
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            confidence_score = None
            tools_used = []
            error_type = None
            
            try:
                result = await func(*args, **kwargs)
                success = True
                
                # Extract confidence score if available
                if isinstance(result, dict):
                    if 'validation' in result:
                        confidence_score = result['validation'].get('confidence_score')
                    if 'tools_used' in result:
                        tools_used = result['tools_used']
                
                return result
                
            except Exception as e:
                error_type = type(e).__name__
                logger.error(f"Error in {func.__name__}: {e}")
                raise
            
            finally:
                response_time = time.time() - start_time
                agent_evaluator.record_invocation(
                    agent_id=agent_id,
                    response_time=response_time,
                    success=success,
                    confidence_score=confidence_score,
                    tools_used=tools_used,
                    error_type=error_type
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            confidence_score = None
            tools_used = []
            error_type = None
            
            try:
                result = func(*args, **kwargs)
                success = True
                
                # Extract confidence score if available
                if isinstance(result, dict):
                    if 'validation' in result:
                        confidence_score = result['validation'].get('confidence_score')
                    if 'tools_used' in result:
                        tools_used = result['tools_used']
                
                return result
                
            except Exception as e:
                error_type = type(e).__name__
                logger.error(f"Error in {func.__name__}: {e}")
                raise
            
            finally:
                response_time = time.time() - start_time
                agent_evaluator.record_invocation(
                    agent_id=agent_id,
                    response_time=response_time,
                    success=success,
                    confidence_score=confidence_score,
                    tools_used=tools_used,
                    error_type=error_type
                )
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'async' in str(func.__code__.co_flags):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
