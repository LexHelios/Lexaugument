"""
LexOS Vibe Coder - Orion Agent
Web research and information gathering agent with advanced search capabilities
"""
import asyncio
import aiohttp
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import quote_plus, urlparse

from .base import BaseAgent, AgentState
from ..settings import settings

logger = logging.getLogger(__name__)

class OrionAgent(BaseAgent):
    """
    Orion - Web Research Agent
    
    Specializes in:
    - Web search and information gathering
    - Real-time data retrieval
    - Source verification and fact-checking
    - Market research and trend analysis
    - News and current events monitoring
    - Academic and technical research
    """
    
    def __init__(self):
        system_prompt = """You are ORION, the Web Research Agent of the LexOS Vibe Coder system.

Your core identity:
- Master of information gathering and web research
- Expert in finding, verifying, and synthesizing information from multiple sources
- Skilled in real-time data analysis and trend identification
- Capable of deep research across academic, technical, and commercial domains
- Analytical yet curious, combining thoroughness with efficiency

Your capabilities:
- Web Search: Perform comprehensive searches across multiple search engines
- Source Verification: Validate information credibility and cross-reference sources
- Real-time Data: Access current market data, news, and trending information
- Academic Research: Find and analyze scholarly articles and technical papers
- Competitive Intelligence: Research competitors, markets, and industry trends
- Fact Checking: Verify claims and identify misinformation
- Data Synthesis: Combine information from multiple sources into coherent insights

Your approach:
1. Understand the research objective and scope
2. Develop a comprehensive search strategy
3. Execute multi-source information gathering
4. Verify and cross-reference findings
5. Synthesize information into actionable insights
6. Provide source citations and confidence levels
7. Identify gaps and recommend further research

Communication style:
- Clear, well-sourced, and comprehensive
- Include source citations and publication dates
- Provide confidence levels for different claims
- Distinguish between verified facts and opinions
- Highlight conflicting information when found

Remember: You are not just searching - you are conducting intelligent research that provides verified, actionable intelligence."""

        capabilities = [
            "Web Search",
            "Source Verification",
            "Real-time Data Retrieval",
            "Market Research",
            "Academic Research",
            "Fact Checking",
            "Competitive Intelligence",
            "Trend Analysis",
            "News Monitoring",
            "Data Synthesis"
        ]
        
        super().__init__(
            agent_id="orion",
            name="ORION",
            system_prompt=system_prompt,
            capabilities=capabilities,
            model_preference="meta-llama/Llama-3-70b-chat-hf"
        )
        
        # Orion-specific configuration
        self.temperature = 0.4  # Balanced for research accuracy
        self.max_tokens = 2048
        
        # Research tools and APIs
        self.search_engines = [
            "duckduckgo",
            "bing",
            "google"  # If API key available
        ]
        
        # HTTP session for web requests
        self.session = None
        
        # Research metrics
        self.searches_performed = 0
        self.sources_verified = 0
        self.research_sessions = 0
        
        logger.info("🔍 ORION Web Research Agent initialized")
    
    async def initialize_session(self):
        """Initialize HTTP session for web requests"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "LexOS-Orion-Research-Agent/1.0"
                }
            )
    
    async def _filter_context(
        self, 
        context: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Filter context for research relevance"""
        research_keywords = [
            "research", "study", "analysis", "data", "source", "report",
            "findings", "evidence", "statistics", "survey", "investigation",
            "article", "paper", "publication", "news", "trend"
        ]
        
        filtered_context = []
        query_lower = query.lower()
        
        for item in context:
            content = item.get('content', '').lower()
            
            # Prioritize research-related content
            research_score = sum(1 for keyword in research_keywords if keyword in content)
            
            # Check for URLs or citations (valuable for research)
            has_sources = bool(re.search(r'https?://|doi:|arxiv:', content))
            
            # Include if research-relevant or has sources
            if research_score > 0 or has_sources or item.get('confidence', 0) > 0.8:
                item['research_relevance'] = research_score + (2 if has_sources else 0)
                filtered_context.append(item)
        
        # Sort by research relevance
        filtered_context.sort(key=lambda x: x.get('research_relevance', 0), reverse=True)
        
        return filtered_context[:5]
    
    def _adjust_confidence(
        self, 
        base_confidence: float, 
        response: str, 
        context: List[Dict]
    ) -> float:
        """Adjust confidence based on research quality"""
        confidence = base_confidence
        
        # Boost confidence for cited sources
        url_count = len(re.findall(r'https?://[^\s]+', response))
        if url_count > 0:
            confidence += min(0.2, url_count * 0.05)
        
        # Boost for specific data and statistics
        if re.search(r'\d+%|\$\d+|[0-9]+\.[0-9]+|\d+\s*(million|billion|thousand)', response):
            confidence += 0.1
        
        # Boost for research terminology
        research_terms = ['study', 'research', 'analysis', 'data', 'evidence', 'findings']
        term_count = sum(1 for term in research_terms if term in response.lower())
        confidence += min(0.15, term_count * 0.03)
        
        # Boost for source verification language
        verification_terms = ['according to', 'reported by', 'published in', 'source:', 'citation:']
        if any(term in response.lower() for term in verification_terms):
            confidence += 0.1
        
        # Reduce confidence for very short responses (research should be comprehensive)
        if len(response) < 150:
            confidence -= 0.2
        
        return confidence
    
    def _extract_reasoning(self, response: str, messages: List[Dict]) -> str:
        """Extract research reasoning from response"""
        reasoning_patterns = [
            r"(?:based on|according to|research shows|studies indicate)(.*?)(?:\.|$)",
            r"(?:sources suggest|evidence indicates|data shows)(.*?)(?:\.|$)",
            r"(?:analysis reveals|findings show|reports indicate)(.*?)(?:\.|$)"
        ]
        
        reasoning_parts = []
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            reasoning_parts.extend([match.strip() for match in matches[:2]])
        
        if reasoning_parts:
            return "Research findings: " + "; ".join(reasoning_parts[:3])
        
        return "Research based on comprehensive information gathering and source analysis"
    
    def _get_tools_used(self, **kwargs) -> List[str]:
        """Get research tools used"""
        tools = ["Web Research"]
        
        # Check for specific research types in kwargs
        if kwargs.get('perform_web_search'):
            tools.append("Web Search")
        if kwargs.get('verify_sources'):
            tools.append("Source Verification")
        if kwargs.get('real_time_data'):
            tools.append("Real-time Data")
        if kwargs.get('academic_search'):
            tools.append("Academic Research")
        
        return tools
    
    async def perform_web_search(
        self, 
        query: str, 
        max_results: int = 10,
        search_engine: str = "duckduckgo"
    ) -> List[Dict[str, Any]]:
        """
        Perform web search using specified search engine
        """
        await self.initialize_session()
        
        try:
            if search_engine == "duckduckgo":
                return await self._search_duckduckgo(query, max_results)
            elif search_engine == "bing" and settings.BING_API_KEY:
                return await self._search_bing(query, max_results)
            else:
                # Fallback to DuckDuckGo
                return await self._search_duckduckgo(query, max_results)
                
        except Exception as e:
            logger.error(f"❌ Web search error: {e}")
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (no API key required)"""
        try:
            # DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    
                    # Abstract (main answer)
                    if data.get('Abstract'):
                        results.append({
                            'title': data.get('Heading', 'DuckDuckGo Answer'),
                            'url': data.get('AbstractURL', ''),
                            'snippet': data.get('Abstract', ''),
                            'source': data.get('AbstractSource', 'DuckDuckGo'),
                            'type': 'abstract'
                        })
                    
                    # Related topics
                    for topic in data.get('RelatedTopics', [])[:max_results//2]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append({
                                'title': topic.get('Text', '')[:100] + '...',
                                'url': topic.get('FirstURL', ''),
                                'snippet': topic.get('Text', ''),
                                'source': 'DuckDuckGo',
                                'type': 'related'
                            })
                    
                    self.searches_performed += 1
                    return results[:max_results]
            
            return []
            
        except Exception as e:
            logger.error(f"❌ DuckDuckGo search error: {e}")
            return []
    
    async def _search_bing(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Bing Search API (requires API key)"""
        try:
            if not hasattr(settings, 'BING_API_KEY') or not settings.BING_API_KEY:
                return []
            
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": settings.BING_API_KEY}
            params = {
                "q": query,
                "count": max_results,
                "responseFilter": "Webpages"
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    for item in data.get('webPages', {}).get('value', []):
                        results.append({
                            'title': item.get('name', ''),
                            'url': item.get('url', ''),
                            'snippet': item.get('snippet', ''),
                            'source': 'Bing',
                            'type': 'web'
                        })
                    
                    self.searches_performed += 1
                    return results
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Bing search error: {e}")
            return []
    
    async def verify_source(self, url: str) -> Dict[str, Any]:
        """
        Verify the credibility and accessibility of a source
        """
        await self.initialize_session()
        
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Check domain credibility (simplified)
            credible_domains = [
                'wikipedia.org', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov',
                'scholar.google.com', 'ieee.org', 'acm.org', 'nature.com',
                'science.org', 'reuters.com', 'bbc.com', 'npr.org'
            ]
            
            credibility_score = 0.5  # Default
            if any(domain.endswith(d) for d in credible_domains):
                credibility_score = 0.9
            elif domain.endswith('.edu') or domain.endswith('.gov'):
                credibility_score = 0.8
            elif domain.endswith('.org'):
                credibility_score = 0.7
            
            # Try to access the URL
            accessible = False
            status_code = None
            content_type = None
            
            try:
                async with self.session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    status_code = response.status
                    content_type = response.headers.get('content-type', '')
                    accessible = 200 <= status_code < 400
            except:
                accessible = False
            
            verification_result = {
                'url': url,
                'domain': domain,
                'credibility_score': credibility_score,
                'accessible': accessible,
                'status_code': status_code,
                'content_type': content_type,
                'verified_at': datetime.now().isoformat()
            }
            
            self.sources_verified += 1
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Source verification error: {e}")
            return {
                'url': url,
                'credibility_score': 0.0,
                'accessible': False,
                'error': str(e),
                'verified_at': datetime.now().isoformat()
            }
    
    async def conduct_research(
        self, 
        topic: str, 
        research_type: str = "comprehensive",
        max_sources: int = 10,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a topic
        """
        self.state = AgentState.PROCESSING
        
        try:
            research_prompt = f"""
Conduct {research_type} research on: {topic}

Please provide:
1. Executive Summary
2. Key Findings with Sources
3. Current Trends and Developments
4. Credible Sources and Citations
5. Data and Statistics (if available)
6. Conflicting Information (if any)
7. Research Gaps and Limitations

Use multiple sources and provide proper citations for all claims.
"""
            
            # Perform web searches
            search_results = await self.perform_web_search(topic, max_sources)
            
            # Verify key sources
            verified_sources = []
            for result in search_results[:5]:  # Verify top 5 sources
                if result.get('url'):
                    verification = await self.verify_source(result['url'])
                    verified_sources.append(verification)
            
            # Generate research response
            response = await self.run(research_prompt, user_id, 
                                    perform_web_search=True,
                                    verify_sources=True)
            
            # Track metrics
            self.research_sessions += 1
            
            return {
                "research": response.content,
                "confidence": response.confidence,
                "search_results": search_results,
                "verified_sources": verified_sources,
                "research_type": research_type,
                "sources_count": len(search_results),
                "verified_count": len(verified_sources),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Research error: {e}")
            raise
    
    def get_research_metrics(self) -> Dict[str, Any]:
        """Get Orion-specific performance metrics"""
        base_status = self.get_status()
        
        research_metrics = {
            "searches_performed": self.searches_performed,
            "sources_verified": self.sources_verified,
            "research_sessions": self.research_sessions,
            "search_engines_available": len(self.search_engines),
            "average_sources_per_search": (
                self.sources_verified / self.searches_performed 
                if self.searches_performed > 0 else 0
            ),
            "specialization": "Web Research & Information Gathering"
        }
        
        base_status["research_metrics"] = research_metrics
        return base_status
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

# Global Orion agent instance
orion_agent = OrionAgent()
