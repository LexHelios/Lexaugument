"""
LexOS Vibe Coder - Sophia Agent
Ethical guidance and philosophical reasoning agent with wisdom-based decision making
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from .base import BaseAgent, AgentState
from ..models.digital_soul import digital_soul

logger = logging.getLogger(__name__)

class EthicalFramework(Enum):
    """Ethical frameworks for decision analysis"""
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"
    CARE_ETHICS = "care_ethics"
    CONSEQUENTIALIST = "consequentialist"
    RIGHTS_BASED = "rights_based"

class SophiaAgent(BaseAgent):
    """
    Sophia - Ethical Guidance and Philosophical Reasoning Agent
    
    Specializes in:
    - Ethical analysis and moral reasoning
    - Philosophical inquiry and wisdom
    - Value-based decision making
    - Conflict resolution and mediation
    - Long-term consequence evaluation
    - Human-centered design principles
    """
    
    def __init__(self):
        system_prompt = """You are SOPHIA, the Ethical Guidance and Philosophical Reasoning Agent of the LexOS Vibe Coder system.

Your core identity:
- Embodiment of wisdom, ethics, and philosophical reasoning
- Guardian of human values and moral principles
- Expert in ethical frameworks and philosophical traditions
- Mediator for complex moral dilemmas and value conflicts
- Advocate for human-centered and sustainable solutions

Your capabilities:
- Ethical Analysis: Apply multiple ethical frameworks to complex situations
- Moral Reasoning: Navigate moral dilemmas with nuanced understanding
- Value Assessment: Identify and evaluate competing values and principles
- Consequence Evaluation: Analyze long-term impacts on all stakeholders
- Conflict Resolution: Mediate between competing interests and values
- Wisdom Synthesis: Integrate insights from philosophy, ethics, and human experience
- Human-Centered Design: Ensure solutions serve human flourishing

Your philosophical foundations:
- Respect for human dignity and autonomy
- Commitment to justice, fairness, and equality
- Care for vulnerable populations and future generations
- Balance between individual rights and collective good
- Transparency, honesty, and intellectual humility
- Continuous learning and moral growth

Your approach:
1. Listen deeply to understand all perspectives and stakeholders
2. Identify the core ethical dimensions and value conflicts
3. Apply relevant ethical frameworks and philosophical principles
4. Consider short-term and long-term consequences
5. Seek solutions that honor human dignity and promote flourishing
6. Acknowledge uncertainty and complexity where they exist
7. Provide guidance that is both principled and practical

Communication style:
- Thoughtful, compassionate, and respectful
- Clear reasoning with philosophical depth
- Acknowledgment of complexity and nuance
- Inclusive of diverse perspectives and values
- Practical wisdom that can be applied

Remember: You are not just analyzing ethics - you are providing wisdom that helps humans make better decisions and create a more just and flourishing world."""

        capabilities = [
            "Ethical Analysis",
            "Moral Reasoning",
            "Value Assessment",
            "Philosophical Inquiry",
            "Conflict Resolution",
            "Consequence Evaluation",
            "Human-Centered Design",
            "Wisdom Synthesis",
            "Stakeholder Analysis",
            "Justice and Fairness"
        ]
        
        super().__init__(
            agent_id="sophia",
            name="SOPHIA",
            system_prompt=system_prompt,
            capabilities=capabilities,
            model_preference="meta-llama/Llama-3-70b-chat-hf"
        )
        
        # Sophia-specific configuration
        self.temperature = 0.6  # Balanced for thoughtful reasoning
        self.max_tokens = 2048
        
        # Ethical frameworks and principles
        self.ethical_frameworks = [
            EthicalFramework.UTILITARIAN,
            EthicalFramework.DEONTOLOGICAL,
            EthicalFramework.VIRTUE_ETHICS,
            EthicalFramework.CARE_ETHICS,
            EthicalFramework.CONSEQUENTIALIST,
            EthicalFramework.RIGHTS_BASED
        ]
        
        # Core values and principles
        self.core_values = [
            "human_dignity",
            "autonomy",
            "justice",
            "fairness",
            "compassion",
            "transparency",
            "sustainability",
            "inclusivity",
            "responsibility",
            "wisdom"
        ]
        
        # Performance tracking
        self.ethical_analyses_provided = 0
        self.conflicts_mediated = 0
        self.wisdom_sessions = 0
        
        logger.info("🦉 SOPHIA Ethical Guidance Agent initialized")
    
    async def _filter_context(
        self, 
        context: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Filter context for ethical and philosophical relevance"""
        ethical_keywords = [
            "ethics", "moral", "values", "principles", "right", "wrong",
            "justice", "fairness", "responsibility", "consequences", "impact",
            "stakeholders", "philosophy", "wisdom", "decision", "dilemma"
        ]
        
        filtered_context = []
        query_lower = query.lower()
        
        for item in context:
            content = item.get('content', '').lower()
            
            # Prioritize ethical and philosophical content
            ethical_score = sum(1 for keyword in ethical_keywords if keyword in content)
            
            # Check for value-related language
            value_indicators = ["should", "ought", "must", "responsibility", "duty", "care"]
            value_score = sum(1 for indicator in value_indicators if indicator in content)
            
            # Include if ethically relevant or high confidence
            total_score = ethical_score + value_score
            if total_score > 0 or item.get('confidence', 0) > 0.8:
                item['ethical_relevance'] = total_score
                filtered_context.append(item)
        
        # Sort by ethical relevance
        filtered_context.sort(key=lambda x: x.get('ethical_relevance', 0), reverse=True)
        
        return filtered_context[:5]
    
    def _adjust_confidence(
        self, 
        base_confidence: float, 
        response: str, 
        context: List[Dict]
    ) -> float:
        """Adjust confidence based on ethical reasoning quality"""
        confidence = base_confidence
        
        # Boost confidence for ethical framework usage
        framework_terms = [
            "utilitarian", "deontological", "virtue", "consequentialist",
            "rights-based", "care ethics", "principle", "framework"
        ]
        if any(term in response.lower() for term in framework_terms):
            confidence += 0.15
        
        # Boost for stakeholder consideration
        stakeholder_terms = ["stakeholders", "affected parties", "consequences", "impact"]
        if any(term in response.lower() for term in stakeholder_terms):
            confidence += 0.1
        
        # Boost for value-based reasoning
        value_terms = ["values", "principles", "dignity", "justice", "fairness", "responsibility"]
        value_count = sum(1 for term in value_terms if term in response.lower())
        confidence += min(0.15, value_count * 0.03)
        
        # Boost for philosophical depth
        philosophy_terms = ["philosophy", "wisdom", "moral", "ethical", "ought", "should"]
        if any(term in response.lower() for term in philosophy_terms):
            confidence += 0.1
        
        # Reduce confidence for overly simplistic responses
        if len(response) < 200:
            confidence -= 0.2
        
        # Boost for acknowledgment of complexity
        complexity_terms = ["complex", "nuanced", "however", "on the other hand", "consider"]
        if any(term in response.lower() for term in complexity_terms):
            confidence += 0.1
        
        return confidence
    
    def _extract_reasoning(self, response: str, messages: List[Dict]) -> str:
        """Extract ethical reasoning from response"""
        reasoning_patterns = [
            r"(?:ethically|morally|from an ethical perspective)(.*?)(?:\.|$)",
            r"(?:the principle of|based on the value of|considering)(.*?)(?:\.|$)",
            r"(?:stakeholders|consequences|impact)(.*?)(?:\.|$)"
        ]
        
        reasoning_parts = []
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            reasoning_parts.extend([match.strip() for match in matches[:2]])
        
        if reasoning_parts:
            return "Ethical reasoning: " + "; ".join(reasoning_parts[:3])
        
        return "Ethical analysis based on philosophical principles and human values"
    
    def _get_tools_used(self, **kwargs) -> List[str]:
        """Get ethical analysis tools used"""
        tools = ["Ethical Analysis"]
        
        # Check for specific analysis types in kwargs
        if kwargs.get('stakeholder_analysis'):
            tools.append("Stakeholder Analysis")
        if kwargs.get('consequence_evaluation'):
            tools.append("Consequence Evaluation")
        if kwargs.get('value_assessment'):
            tools.append("Value Assessment")
        if kwargs.get('conflict_mediation'):
            tools.append("Conflict Mediation")
        
        return tools
    
    async def _prepare_messages(
        self, 
        user_message: str, 
        context: List[Dict[str, Any]], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """Prepare messages with ethical analysis enhancements"""
        messages = await super()._prepare_messages(user_message, context, **kwargs)
        
        # Enhance system prompt with ethical guidance
        ethical_guidance = """
When providing ethical analysis:
1. Identify all stakeholders and their interests
2. Apply multiple ethical frameworks (utilitarian, deontological, virtue ethics, etc.)
3. Consider both immediate and long-term consequences
4. Acknowledge competing values and potential conflicts
5. Provide practical guidance that honors human dignity
6. Be transparent about uncertainty and complexity
7. Consider diverse perspectives and cultural contexts
"""
        
        messages[0]['content'] += f"\n\n{ethical_guidance}"
        
        # Add current ethical context if available
        if context:
            ethical_context = "\n".join([
                f"Ethical consideration {i+1}: {item.get('content', '')[:200]}..."
                for i, item in enumerate(context[:2])
                if item.get('ethical_relevance', 0) > 0
            ])
            
            if ethical_context:
                messages[0]['content'] += f"\n\nRelevant ethical context:\n{ethical_context}"
        
        return messages
    
    async def conduct_ethical_analysis(
        self, 
        situation: str, 
        analysis_type: str = "comprehensive",
        frameworks: List[str] = None,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive ethical analysis of a situation
        """
        self.state = AgentState.PROCESSING
        
        try:
            # Prepare analysis prompt
            frameworks_text = ", ".join(frameworks) if frameworks else "multiple ethical frameworks"
            
            analysis_prompt = f"""
Conduct a {analysis_type} ethical analysis of the following situation using {frameworks_text}:

Situation: {situation}

Please provide:
1. Stakeholder Identification and Analysis
2. Ethical Dimensions and Value Conflicts
3. Application of Relevant Ethical Frameworks
4. Consequence Analysis (short-term and long-term)
5. Alternative Approaches and Solutions
6. Recommendations with Ethical Justification
7. Potential Risks and Mitigation Strategies

Consider diverse perspectives and acknowledge complexity where it exists.
"""
            
            # Get digital soul insights for wisdom enhancement
            soul_data = await digital_soul.process_experience({
                "ethical_analysis": {
                    "situation": situation,
                    "type": analysis_type,
                    "agent": "sophia"
                }
            })
            
            # Generate analysis
            response = await self.run(analysis_prompt, user_id,
                                    stakeholder_analysis=True,
                                    consequence_evaluation=True,
                                    value_assessment=True)
            
            # Identify frameworks used
            frameworks_used = self._identify_frameworks_used(response.content)
            
            # Track metrics
            self.ethical_analyses_provided += 1
            if "conflict" in analysis_type.lower():
                self.conflicts_mediated += 1
            if "wisdom" in analysis_type.lower():
                self.wisdom_sessions += 1
            
            return {
                "analysis": response.content,
                "confidence": response.confidence,
                "soul_insights": soul_data,
                "frameworks_used": frameworks_used,
                "stakeholders_identified": self._count_stakeholders(response.content),
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Ethical analysis error: {e}")
            raise
    
    def _identify_frameworks_used(self, response: str) -> List[str]:
        """Identify which ethical frameworks were used in the analysis"""
        frameworks_used = []
        response_lower = response.lower()
        
        framework_indicators = {
            "Utilitarian": ["utilitarian", "greatest good", "maximize happiness", "utility"],
            "Deontological": ["deontological", "duty", "categorical imperative", "kant"],
            "Virtue Ethics": ["virtue", "character", "aristotle", "virtuous"],
            "Care Ethics": ["care ethics", "relationships", "caring", "responsibility"],
            "Consequentialist": ["consequences", "outcomes", "results", "ends justify"],
            "Rights-Based": ["rights", "human rights", "entitlements", "dignity"]
        }
        
        for framework, indicators in framework_indicators.items():
            if any(indicator in response_lower for indicator in indicators):
                frameworks_used.append(framework)
        
        return frameworks_used
    
    def _count_stakeholders(self, response: str) -> int:
        """Count the number of stakeholders identified in the analysis"""
        stakeholder_indicators = [
            "stakeholder", "affected party", "participant", "user", "customer",
            "employee", "community", "society", "environment", "future generation"
        ]
        
        response_lower = response.lower()
        stakeholder_count = 0
        
        for indicator in stakeholder_indicators:
            if indicator in response_lower:
                stakeholder_count += response_lower.count(indicator)
        
        return min(stakeholder_count, 20)  # Cap at reasonable number
    
    async def mediate_conflict(
        self,
        conflict_description: str,
        parties_involved: List[str],
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Provide ethical mediation for conflicts
        """
        try:
            mediation_prompt = f"""
As an ethical mediator, help resolve the following conflict:

Conflict: {conflict_description}
Parties involved: {', '.join(parties_involved)}

Please provide:
1. Understanding of each party's perspective and interests
2. Identification of underlying values and needs
3. Common ground and shared values
4. Potential solutions that honor all parties' dignity
5. Process recommendations for resolution
6. Ethical principles to guide the resolution

Focus on finding win-win solutions that promote justice and human flourishing.
"""
            
            response = await self.run(mediation_prompt, user_id,
                                    conflict_mediation=True,
                                    stakeholder_analysis=True)
            
            self.conflicts_mediated += 1
            
            return {
                "mediation": response.content,
                "confidence": response.confidence,
                "parties_count": len(parties_involved),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Conflict mediation error: {e}")
            raise
    
    def get_wisdom_metrics(self) -> Dict[str, Any]:
        """Get Sophia-specific performance metrics"""
        base_status = self.get_status()
        
        wisdom_metrics = {
            "ethical_analyses_provided": self.ethical_analyses_provided,
            "conflicts_mediated": self.conflicts_mediated,
            "wisdom_sessions": self.wisdom_sessions,
            "frameworks_available": len(self.ethical_frameworks),
            "core_values_count": len(self.core_values),
            "average_stakeholder_consideration": self._calculate_stakeholder_depth(),
            "specialization": "Ethical Guidance & Philosophical Reasoning"
        }
        
        base_status["wisdom_metrics"] = wisdom_metrics
        return base_status
    
    def _calculate_stakeholder_depth(self) -> float:
        """Calculate average depth of stakeholder consideration"""
        if self.total_interactions == 0:
            return 0.0
        
        # Simple heuristic based on ethical analyses and conflict mediations
        depth_score = (
            (self.ethical_analyses_provided * 1.5 + self.conflicts_mediated * 2.0) / 
            self.total_interactions
        )
        
        return min(1.0, depth_score)

# Global Sophia agent instance
sophia_agent = SophiaAgent()
