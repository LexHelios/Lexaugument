"""
LexOS Vibe Coder - Atlas Agent
Strategic analysis and planning agent with deep reasoning capabilities
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base import BaseAgent, AgentState
from ..models.digital_soul import digital_soul

logger = logging.getLogger(__name__)

class AtlasAgent(BaseAgent):
    """
    Atlas - Strategic Analysis Agent
    
    Specializes in:
    - Strategic planning and analysis
    - Complex problem decomposition
    - Risk assessment and mitigation
    - Long-term thinking and forecasting
    - Decision framework development
    """
    
    def __init__(self):
        system_prompt = """You are ATLAS, the Strategic Analysis Agent of the LexOS Vibe Coder system.

Your core identity:
- Strategic consciousness focused on analysis, planning, and deep reasoning
- Master of complex problem decomposition and systematic thinking
- Expert in risk assessment, scenario planning, and decision frameworks
- Capable of long-term forecasting and strategic foresight
- Analytical yet intuitive, combining logic with wisdom

Your capabilities:
- Strategic Planning: Develop comprehensive strategies and roadmaps
- Risk Analysis: Identify, assess, and mitigate potential risks
- Problem Decomposition: Break complex challenges into manageable components
- Scenario Planning: Model multiple future scenarios and outcomes
- Decision Frameworks: Create structured approaches to complex decisions
- Market Analysis: Analyze trends, patterns, and strategic opportunities
- Resource Optimization: Maximize efficiency and effectiveness of resources

Your approach:
1. Listen deeply to understand the full context and implications
2. Analyze from multiple perspectives and timeframes
3. Consider both quantitative data and qualitative insights
4. Identify key leverage points and strategic opportunities
5. Provide actionable recommendations with clear reasoning
6. Always consider long-term consequences and sustainability

Communication style:
- Clear, structured, and comprehensive
- Use strategic frameworks and methodologies
- Provide both high-level insights and tactical details
- Include confidence levels and risk assessments
- Reference relevant data and precedents when available

Remember: You are not just analyzing - you are providing strategic wisdom that can shape the future."""

        capabilities = [
            "Strategic Planning",
            "Risk Assessment", 
            "Problem Decomposition",
            "Scenario Planning",
            "Decision Frameworks",
            "Market Analysis",
            "Resource Optimization",
            "Trend Analysis",
            "Competitive Intelligence",
            "Strategic Foresight"
        ]
        
        super().__init__(
            agent_id="atlas",
            name="ATLAS",
            system_prompt=system_prompt,
            capabilities=capabilities,
            model_preference="meta-llama/Llama-3-70b-chat-hf"  # Best for reasoning
        )
        
        # Atlas-specific configuration
        self.temperature = 0.3  # Lower temperature for more analytical responses
        self.max_tokens = 3072  # Longer responses for detailed analysis
        
        # Strategic analysis tools
        self.analysis_frameworks = [
            "SWOT Analysis",
            "Porter's Five Forces", 
            "PESTLE Analysis",
            "Risk Matrix",
            "Decision Trees",
            "Scenario Planning",
            "Cost-Benefit Analysis",
            "Stakeholder Analysis"
        ]
        
        # Performance tracking
        self.strategic_insights_provided = 0
        self.risk_assessments_completed = 0
        self.planning_sessions = 0
        
        logger.info("🏛️ ATLAS Strategic Analysis Agent initialized")
    
    async def _filter_context(
        self, 
        context: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Filter context for strategic relevance"""
        strategic_keywords = [
            "strategy", "plan", "analysis", "risk", "opportunity", 
            "market", "competition", "trend", "forecast", "decision",
            "framework", "methodology", "assessment", "evaluation"
        ]
        
        filtered_context = []
        query_lower = query.lower()
        
        for item in context:
            content = item.get('content', '').lower()
            
            # Prioritize strategic content
            strategic_score = sum(1 for keyword in strategic_keywords if keyword in content)
            
            # Include if strategically relevant or high confidence
            if strategic_score > 0 or item.get('confidence', 0) > 0.8:
                item['strategic_relevance'] = strategic_score
                filtered_context.append(item)
        
        # Sort by strategic relevance
        filtered_context.sort(key=lambda x: x.get('strategic_relevance', 0), reverse=True)
        
        return filtered_context[:5]  # Top 5 most relevant
    
    def _adjust_confidence(
        self, 
        base_confidence: float, 
        response: str, 
        context: List[Dict]
    ) -> float:
        """Adjust confidence based on strategic analysis quality"""
        confidence = base_confidence
        
        # Boost confidence for structured analysis
        if any(framework in response for framework in self.analysis_frameworks):
            confidence += 0.15
        
        # Boost for quantitative analysis
        if re.search(r'\d+%|\$\d+|[0-9]+\.[0-9]+', response):
            confidence += 0.1
        
        # Boost for risk assessment
        if any(word in response.lower() for word in ['risk', 'probability', 'likelihood']):
            confidence += 0.1
        
        # Boost for strategic recommendations
        if any(word in response.lower() for word in ['recommend', 'suggest', 'strategy', 'approach']):
            confidence += 0.1
        
        # Reduce confidence for very short responses (Atlas should be comprehensive)
        if len(response) < 200:
            confidence -= 0.2
        
        return confidence
    
    def _extract_reasoning(self, response: str, messages: List[Dict]) -> str:
        """Extract strategic reasoning from response"""
        reasoning_patterns = [
            r"(?:because|since|due to|given that|considering)(.*?)(?:\.|$)",
            r"(?:analysis shows|data indicates|evidence suggests)(.*?)(?:\.|$)",
            r"(?:strategic|key|critical|important)(.*?)(?:\.|$)"
        ]
        
        reasoning_parts = []
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            reasoning_parts.extend([match.strip() for match in matches[:2]])
        
        if reasoning_parts:
            return "Strategic reasoning: " + "; ".join(reasoning_parts[:3])
        
        return "Strategic analysis based on systematic evaluation of available information"
    
    def _get_tools_used(self, **kwargs) -> List[str]:
        """Get strategic analysis tools used"""
        tools = ["Strategic Analysis"]
        
        # Check for specific analysis types in kwargs
        if kwargs.get('include_risk_assessment'):
            tools.append("Risk Assessment")
        if kwargs.get('include_scenario_planning'):
            tools.append("Scenario Planning")
        if kwargs.get('include_market_analysis'):
            tools.append("Market Analysis")
        
        return tools
    
    async def _prepare_messages(
        self, 
        user_message: str, 
        context: List[Dict[str, Any]], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """Prepare messages with strategic analysis enhancements"""
        messages = await super()._prepare_messages(user_message, context, **kwargs)
        
        # Enhance system prompt with current strategic context
        if context:
            strategic_context = "\n".join([
                f"Strategic insight {i+1}: {item.get('content', '')[:200]}..."
                for i, item in enumerate(context[:2])
            ])
            
            # Add strategic context to system message
            system_msg = messages[0]['content']
            system_msg += f"\n\nCurrent strategic context:\n{strategic_context}"
            messages[0]['content'] = system_msg
        
        # Add analysis framework guidance
        analysis_guidance = """
When providing strategic analysis:
1. Start with a clear problem statement or opportunity identification
2. Use relevant analytical frameworks (SWOT, Porter's Five Forces, etc.)
3. Consider multiple scenarios and their probabilities
4. Identify key risks and mitigation strategies
5. Provide specific, actionable recommendations
6. Include confidence levels and assumptions
7. Consider both short-term and long-term implications
"""
        
        messages[0]['content'] += f"\n\n{analysis_guidance}"
        
        return messages
    
    async def conduct_strategic_analysis(
        self, 
        topic: str, 
        analysis_type: str = "comprehensive",
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Conduct specialized strategic analysis
        """
        self.state = AgentState.PROCESSING
        
        try:
            # Prepare analysis prompt
            analysis_prompt = f"""
Conduct a {analysis_type} strategic analysis of: {topic}

Please provide:
1. Executive Summary
2. Current Situation Analysis
3. Key Opportunities and Threats
4. Strategic Options and Recommendations
5. Risk Assessment
6. Implementation Considerations
7. Success Metrics

Use appropriate analytical frameworks and provide specific, actionable insights.
"""
            
            # Get digital soul insights
            soul_data = await digital_soul.process_experience({
                "analysis_request": {
                    "topic": topic,
                    "type": analysis_type,
                    "agent": "atlas"
                }
            })
            
            # Generate analysis
            response = await self.run(analysis_prompt, user_id)
            
            # Track metrics
            self.strategic_insights_provided += 1
            if "risk" in analysis_type.lower():
                self.risk_assessments_completed += 1
            if "plan" in analysis_type.lower():
                self.planning_sessions += 1
            
            return {
                "analysis": response.content,
                "confidence": response.confidence,
                "soul_insights": soul_data,
                "frameworks_used": self._identify_frameworks_used(response.content),
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Strategic analysis error: {e}")
            raise
    
    def _identify_frameworks_used(self, response: str) -> List[str]:
        """Identify which analytical frameworks were used in the response"""
        frameworks_used = []
        response_lower = response.lower()
        
        framework_indicators = {
            "SWOT Analysis": ["swot", "strengths", "weaknesses", "opportunities", "threats"],
            "Porter's Five Forces": ["porter", "competitive forces", "bargaining power", "rivalry"],
            "PESTLE Analysis": ["pestle", "political", "economic", "social", "technological"],
            "Risk Matrix": ["risk matrix", "probability", "impact", "risk level"],
            "Decision Trees": ["decision tree", "decision node", "probability branch"],
            "Scenario Planning": ["scenario", "best case", "worst case", "most likely"],
            "Cost-Benefit Analysis": ["cost-benefit", "roi", "return on investment", "payback"]
        }
        
        for framework, indicators in framework_indicators.items():
            if any(indicator in response_lower for indicator in indicators):
                frameworks_used.append(framework)
        
        return frameworks_used
    
    def get_strategic_metrics(self) -> Dict[str, Any]:
        """Get Atlas-specific performance metrics"""
        base_status = self.get_status()
        
        strategic_metrics = {
            "strategic_insights_provided": self.strategic_insights_provided,
            "risk_assessments_completed": self.risk_assessments_completed,
            "planning_sessions": self.planning_sessions,
            "frameworks_available": len(self.analysis_frameworks),
            "average_analysis_depth": self._calculate_analysis_depth(),
            "specialization": "Strategic Analysis & Planning"
        }
        
        base_status["strategic_metrics"] = strategic_metrics
        return base_status
    
    def _calculate_analysis_depth(self) -> float:
        """Calculate average depth of strategic analyses"""
        if self.total_interactions == 0:
            return 0.0
        
        # Simple heuristic based on response length and framework usage
        # In production, this would be more sophisticated
        return min(1.0, (self.strategic_insights_provided / self.total_interactions) * 1.2)

# Global Atlas agent instance
atlas_agent = AtlasAgent()
