"""Evaluation Dashboard for Agent Performance Monitoring."""

import json
from datetime import datetime
from typing import Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .agent_evaluation import agent_evaluator


class EvaluationDashboard:
    """Dashboard for visualizing agent evaluation metrics."""
    
    def __init__(self):
        self.evaluator = agent_evaluator
    
    def render_dashboard(self) -> None:
        """Render the complete evaluation dashboard."""
        st.title("🔍 Agent Performance Dashboard")
        st.markdown("Monitor and evaluate the performance of investment advisory agents.")
        
        # System overview
        self._render_system_overview()
        
        # Agent-specific metrics
        self._render_agent_metrics()
        
        # Performance trends
        self._render_performance_trends()
        
        # Recommendations
        self._render_recommendations()
    
    def _render_system_overview(self) -> None:
        """Render system-wide overview metrics."""
        st.header("📊 System Overview")
        
        system_report = self.evaluator.generate_system_report()
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System Health",
                f"{system_report.system_health_score:.1%}",
                delta=f"{'🟢' if system_report.system_health_score >= 0.8 else '🟡' if system_report.system_health_score >= 0.6 else '🔴'}"
            )
        
        with col2:
            st.metric(
                "Active Agents",
                system_report.active_agents,
                delta=f"of {system_report.total_agents} total"
            )
        
        with col3:
            success_rate = system_report.successful_queries / system_report.total_queries if system_report.total_queries > 0 else 0
            st.metric(
                "Success Rate",
                f"{success_rate:.1%}",
                delta=f"{system_report.successful_queries} of {system_report.total_queries}"
            )
        
        with col4:
            st.metric(
                "Avg Response Time",
                f"{system_report.average_system_response_time:.1f}s",
                delta=f"{'🟢' if system_report.average_system_response_time <= 5 else '🟡' if system_report.average_system_response_time <= 10 else '🔴'}"
            )
        
        # System uptime
        st.info(f"🕐 System Uptime: {system_report.system_uptime:.1f} hours")
        
        # System recommendations
        if system_report.recommendations:
            st.subheader("🎯 System Recommendations")
            for rec in system_report.recommendations:
                st.warning(f"⚠️ {rec}")
    
    def _render_agent_metrics(self) -> None:
        """Render individual agent metrics."""
        st.header("🤖 Agent Performance")
        
        # Get all agents with metrics
        agents_with_metrics = [
            agent_id for agent_id, metrics in self.evaluator.metrics_store.items()
            if metrics.total_invocations > 0
        ]
        
        if not agents_with_metrics:
            st.warning("No agent metrics available yet.")
            return
        
        # Agent selector
        selected_agent = st.selectbox(
            "Select Agent to Analyze",
            agents_with_metrics,
            format_func=lambda x: x.replace("-", " ").title()
        )
        
        if selected_agent:
            evaluation = self.evaluator.evaluate_agent(selected_agent)
            metrics = self.evaluator.get_agent_metrics(selected_agent)
            
            # Performance scores
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Overall Score",
                    f"{evaluation.overall_score:.1%}",
                    delta=self._get_performance_indicator(evaluation.overall_score)
                )
            
            with col2:
                st.metric(
                    "Accuracy",
                    f"{evaluation.accuracy_score:.1%}",
                    delta=self._get_performance_indicator(evaluation.accuracy_score)
                )
            
            with col3:
                st.metric(
                    "Reliability",
                    f"{evaluation.reliability_score:.1%}",
                    delta=self._get_performance_indicator(evaluation.reliability_score)
                )
            
            with col4:
                st.metric(
                    "Efficiency",
                    f"{evaluation.efficiency_score:.1%}",
                    delta=self._get_performance_indicator(evaluation.efficiency_score)
                )
            
            # Detailed metrics
            st.subheader("📈 Detailed Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Usage Statistics:**")
                st.write(f"• Total Invocations: {metrics.total_invocations}")
                st.write(f"• Success Rate: {evaluation.performance_metrics['success_rate']:.1%}")
                st.write(f"• Average Response Time: {evaluation.performance_metrics['average_response_time']:.2f}s")
                st.write(f"• Average Confidence: {evaluation.performance_metrics['average_confidence']:.1%}")
            
            with col2:
                if metrics.tool_usage:
                    st.write("**Tool Usage:**")
                    for tool, count in sorted(metrics.tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"• {tool}: {count} times")
            
            # Performance charts
            self._render_agent_charts(selected_agent, metrics, evaluation)
            
            # Strengths and weaknesses
            col1, col2 = st.columns(2)
            
            with col1:
                if evaluation.strengths:
                    st.subheader("💪 Strengths")
                    for strength in evaluation.strengths:
                        st.success(f"✅ {strength}")
            
            with col2:
                if evaluation.weaknesses:
                    st.subheader("⚠️ Areas for Improvement")
                    for weakness in evaluation.weaknesses:
                        st.warning(f"🔸 {weakness}")
            
            # Recommendations
            if evaluation.recommendations:
                st.subheader("🎯 Recommendations")
                for rec in evaluation.recommendations:
                    st.info(f"💡 {rec}")
    
    def _render_agent_charts(self, agent_id: str, metrics, evaluation) -> None:
        """Render charts for agent performance."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance scores radar chart
            categories = ['Accuracy', 'Reliability', 'Efficiency']
            scores = [
                evaluation.accuracy_score,
                evaluation.reliability_score,
                evaluation.efficiency_score
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name=agent_id.replace("-", " ").title()
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Performance Scores"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tool usage pie chart
            if metrics.tool_usage:
                fig = px.pie(
                    values=list(metrics.tool_usage.values()),
                    names=list(metrics.tool_usage.keys()),
                    title="Tool Usage Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No tool usage data available")
        
        # Confidence score trend
        if metrics.confidence_scores:
            st.subheader("📊 Confidence Score Trend")
            
            fig = px.line(
                x=list(range(len(metrics.confidence_scores))),
                y=metrics.confidence_scores,
                title="Confidence Scores Over Time",
                labels={'x': 'Invocation Number', 'y': 'Confidence Score'}
            )
            
            # Add average line
            avg_confidence = sum(metrics.confidence_scores) / len(metrics.confidence_scores)
            fig.add_hline(y=avg_confidence, line_dash="dash", 
                         annotation_text=f"Average: {avg_confidence:.2f}")
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_trends(self) -> None:
        """Render system-wide performance trends."""
        st.header("📈 Performance Trends")
        
        # Get evaluation history
        if not self.evaluator.evaluation_history:
            st.info("No historical evaluation data available yet.")
            return
        
        # Group evaluations by agent
        agent_trends = {}
        for evaluation in self.evaluator.evaluation_history[-50:]:  # Last 50 evaluations
            if evaluation.agent_id not in agent_trends:
                agent_trends[evaluation.agent_id] = {
                    'timestamps': [],
                    'overall_scores': [],
                    'accuracy_scores': [],
                    'reliability_scores': [],
                    'efficiency_scores': []
                }
            
            agent_trends[evaluation.agent_id]['timestamps'].append(evaluation.evaluation_timestamp)
            agent_trends[evaluation.agent_id]['overall_scores'].append(evaluation.overall_score)
            agent_trends[evaluation.agent_id]['accuracy_scores'].append(evaluation.accuracy_score)
            agent_trends[evaluation.agent_id]['reliability_scores'].append(evaluation.reliability_score)
            agent_trends[evaluation.agent_id]['efficiency_scores'].append(evaluation.efficiency_score)
        
        # Create trend charts
        if agent_trends:
            # Overall performance trend
            fig = go.Figure()
            
            for agent_id, trends in agent_trends.items():
                fig.add_trace(go.Scatter(
                    x=trends['timestamps'],
                    y=trends['overall_scores'],
                    mode='lines+markers',
                    name=agent_id.replace("-", " ").title(),
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Overall Performance Trends",
                xaxis_title="Time",
                yaxis_title="Performance Score",
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison
            st.subheader("🔄 Agent Performance Comparison")
            
            comparison_data = []
            for agent_id, trends in agent_trends.items():
                if trends['overall_scores']:
                    comparison_data.append({
                        'Agent': agent_id.replace("-", " ").title(),
                        'Latest Score': trends['overall_scores'][-1],
                        'Average Score': sum(trends['overall_scores']) / len(trends['overall_scores']),
                        'Evaluations': len(trends['overall_scores'])
                    })
            
            if comparison_data:
                import pandas as pd
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
    
    def _render_recommendations(self) -> None:
        """Render system recommendations and insights."""
        st.header("🎯 Insights & Recommendations")
        
        system_report = self.evaluator.generate_system_report()
        
        # Performance insights
        if system_report.agent_evaluations:
            best_performer = max(system_report.agent_evaluations, key=lambda x: x.overall_score)
            worst_performer = min(system_report.agent_evaluations, key=lambda x: x.overall_score)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"🏆 **Best Performer:** {best_performer.agent_id.replace('-', ' ').title()}")
                st.write(f"Score: {best_performer.overall_score:.1%}")
                if best_performer.strengths:
                    st.write("Key Strengths:")
                    for strength in best_performer.strengths[:3]:
                        st.write(f"• {strength}")
            
            with col2:
                st.warning(f"⚠️ **Needs Attention:** {worst_performer.agent_id.replace('-', ' ').title()}")
                st.write(f"Score: {worst_performer.overall_score:.1%}")
                if worst_performer.recommendations:
                    st.write("Priority Actions:")
                    for rec in worst_performer.recommendations[:3]:
                        st.write(f"• {rec}")
        
        # Export functionality
        st.subheader("📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Metrics (JSON)"):
                metrics_json = self.evaluator.export_metrics('json')
                st.download_button(
                    label="Download Metrics",
                    data=metrics_json,
                    file_name=f"agent_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Generate Report"):
                report = system_report
                report_text = f"""
# Agent Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Overview
- System Health Score: {report.system_health_score:.1%}
- Active Agents: {report.active_agents}/{report.total_agents}
- Total Queries: {report.total_queries}
- Success Rate: {report.successful_queries/report.total_queries:.1%}
- Average Response Time: {report.average_system_response_time:.2f}s

## Agent Evaluations
"""
                for eval in report.agent_evaluations:
                    report_text += f"""
### {eval.agent_id.replace('-', ' ').title()}
- Overall Score: {eval.overall_score:.1%}
- Accuracy: {eval.accuracy_score:.1%}
- Reliability: {eval.reliability_score:.1%}
- Efficiency: {eval.efficiency_score:.1%}
"""
                
                st.download_button(
                    label="Download Report",
                    data=report_text,
                    file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
    def _get_performance_indicator(self, score: float) -> str:
        """Get performance indicator emoji based on score."""
        if score >= 0.8:
            return "🟢"
        elif score >= 0.6:
            return "🟡"
        else:
            return "🔴"


# Dashboard instance
evaluation_dashboard = EvaluationDashboard()
