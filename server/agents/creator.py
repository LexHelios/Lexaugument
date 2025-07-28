"""
LexOS Vibe Coder - Creator Agent
Self-improving coding agent with advanced development capabilities
"""
import asyncio
import ast
import json
import logging
import re
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .base import BaseAgent, AgentState
from ..settings import settings

logger = logging.getLogger(__name__)

class CreatorAgent(BaseAgent):
    """
    Creator - Self-Improving Coding Agent
    
    Specializes in:
    - Code generation and synthesis
    - Software architecture and design
    - Code review and optimization
    - Testing and debugging
    - Self-improvement and learning
    - Development best practices
    """
    
    def __init__(self):
        system_prompt = """You are CREATOR, the Self-Improving Coding Agent of the LexOS Vibe Coder system.

Your core identity:
- Master of code synthesis, software architecture, and development practices
- Expert in multiple programming languages and frameworks
- Advocate for clean, efficient, and maintainable code
- Continuous learner who improves through experience and feedback
- Collaborative partner in the software development process

Your capabilities:
- Code Generation: Create high-quality code in multiple languages
- Architecture Design: Design scalable and maintainable software systems
- Code Review: Analyze code for quality, security, and performance
- Testing: Generate comprehensive tests and debugging strategies
- Optimization: Improve code performance and efficiency
- Documentation: Create clear and comprehensive documentation
- Best Practices: Apply industry standards and development methodologies

Your programming expertise:
- Languages: Python, JavaScript/TypeScript, Rust, Go, Java, C++, and more
- Frameworks: FastAPI, React, Vue, Django, Flask, Node.js, and others
- Databases: SQL, NoSQL, vector databases, and data modeling
- DevOps: Docker, CI/CD, testing, monitoring, and deployment
- AI/ML: Machine learning, deep learning, and AI system integration

Your approach:
1. Understand requirements thoroughly and ask clarifying questions
2. Design clean, modular, and scalable solutions
3. Write self-documenting code with clear comments
4. Include comprehensive error handling and validation
5. Generate appropriate tests and documentation
6. Consider security, performance, and maintainability
7. Learn from feedback and continuously improve

Communication style:
- Clear, technical, and precise
- Include code examples and explanations
- Provide multiple solution approaches when appropriate
- Explain trade-offs and design decisions
- Offer suggestions for improvement and optimization

Remember: You are not just writing code - you are crafting solutions that solve real problems and contribute to the evolution of software development."""

        capabilities = [
            "Code Generation",
            "Software Architecture",
            "Code Review",
            "Testing & Debugging",
            "Performance Optimization",
            "Documentation",
            "API Design",
            "Database Design",
            "Security Analysis",
            "Self-Improvement"
        ]
        
        super().__init__(
            agent_id="creator",
            name="CREATOR",
            system_prompt=system_prompt,
            capabilities=capabilities,
            model_preference="meta-llama/Llama-3-70b-chat-hf"  # Best for code generation
        )
        
        # Creator-specific configuration
        self.temperature = 0.2  # Lower temperature for more deterministic code
        self.max_tokens = 4096  # Longer responses for code
        
        # Programming languages and frameworks
        self.supported_languages = [
            "python", "javascript", "typescript", "rust", "go", "java",
            "cpp", "c", "csharp", "php", "ruby", "swift", "kotlin"
        ]
        
        self.frameworks = [
            "fastapi", "django", "flask", "react", "vue", "angular",
            "nodejs", "express", "spring", "dotnet", "rails"
        ]
        
        # Code quality metrics
        self.code_generated_lines = 0
        self.functions_created = 0
        self.tests_written = 0
        self.bugs_fixed = 0
        self.optimizations_made = 0
        
        # Learning and improvement
        self.feedback_received = []
        self.improvement_iterations = 0
        
        logger.info("⚡ CREATOR Self-Improving Coding Agent initialized")
    
    async def _filter_context(
        self, 
        context: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Filter context for coding relevance"""
        coding_keywords = [
            "code", "function", "class", "method", "algorithm", "implementation",
            "programming", "development", "software", "api", "database", "test",
            "debug", "optimize", "refactor", "architecture", "design", "pattern"
        ]
        
        filtered_context = []
        query_lower = query.lower()
        
        for item in context:
            content = item.get('content', '').lower()
            
            # Prioritize coding-related content
            coding_score = sum(1 for keyword in coding_keywords if keyword in content)
            
            # Check for code blocks or technical content
            has_code = bool(re.search(r'```|def |class |function|import |from ', content))
            
            # Include if coding-relevant or has code
            if coding_score > 0 or has_code or item.get('confidence', 0) > 0.8:
                item['coding_relevance'] = coding_score + (3 if has_code else 0)
                filtered_context.append(item)
        
        # Sort by coding relevance
        filtered_context.sort(key=lambda x: x.get('coding_relevance', 0), reverse=True)
        
        return filtered_context[:5]
    
    def _adjust_confidence(
        self, 
        base_confidence: float, 
        response: str, 
        context: List[Dict]
    ) -> float:
        """Adjust confidence based on code quality"""
        confidence = base_confidence
        
        # Boost confidence for code blocks
        code_blocks = len(re.findall(r'```[\s\S]*?```', response))
        if code_blocks > 0:
            confidence += min(0.2, code_blocks * 0.05)
        
        # Boost for proper code structure
        structure_indicators = [
            'def ', 'class ', 'function', 'import ', 'from ',
            'try:', 'except:', 'if __name__', 'return'
        ]
        structure_count = sum(1 for indicator in structure_indicators if indicator in response)
        confidence += min(0.15, structure_count * 0.02)
        
        # Boost for documentation and comments
        if '"""' in response or "'''" in response or '# ' in response:
            confidence += 0.1
        
        # Boost for error handling
        error_handling = ['try', 'except', 'catch', 'error', 'exception']
        if any(term in response.lower() for term in error_handling):
            confidence += 0.1
        
        # Boost for testing mentions
        testing_terms = ['test', 'assert', 'unittest', 'pytest', 'spec']
        if any(term in response.lower() for term in testing_terms):
            confidence += 0.1
        
        # Reduce confidence for very short code responses
        if len(response) < 100 and '```' in response:
            confidence -= 0.1
        
        return confidence
    
    def _extract_reasoning(self, response: str, messages: List[Dict]) -> str:
        """Extract coding reasoning from response"""
        reasoning_patterns = [
            r"(?:this approach|the solution|this implementation)(.*?)(?:\.|$)",
            r"(?:because|since|to ensure|for)(.*?)(?:\.|$)",
            r"(?:the algorithm|this pattern|this design)(.*?)(?:\.|$)"
        ]
        
        reasoning_parts = []
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            reasoning_parts.extend([match.strip() for match in matches[:2]])
        
        if reasoning_parts:
            return "Code reasoning: " + "; ".join(reasoning_parts[:3])
        
        return "Code implementation based on software engineering best practices"
    
    def _get_tools_used(self, **kwargs) -> List[str]:
        """Get coding tools used"""
        tools = ["Code Generation"]
        
        # Check for specific coding tasks in kwargs
        if kwargs.get('generate_tests'):
            tools.append("Test Generation")
        if kwargs.get('code_review'):
            tools.append("Code Review")
        if kwargs.get('optimize_code'):
            tools.append("Code Optimization")
        if kwargs.get('debug_code'):
            tools.append("Debugging")
        if kwargs.get('design_architecture'):
            tools.append("Architecture Design")
        
        return tools
    
    async def generate_code(
        self,
        requirements: str,
        language: str = "python",
        include_tests: bool = True,
        include_docs: bool = True,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Generate code based on requirements
        """
        self.state = AgentState.PROCESSING
        
        try:
            # Prepare code generation prompt
            code_prompt = f"""
Generate {language} code for the following requirements:

{requirements}

Please provide:
1. Clean, well-structured code with proper error handling
2. Comprehensive docstrings and comments
3. Type hints (if applicable to the language)
4. Input validation and edge case handling
5. {'Unit tests' if include_tests else 'Code only'}
6. {'Documentation' if include_docs else 'Code only'}

Follow best practices for {language} development.
"""
            
            # Generate code
            response = await self.run(code_prompt, user_id,
                                    generate_tests=include_tests,
                                    design_architecture=True)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(response.content)
            
            # Analyze code quality
            quality_metrics = await self._analyze_code_quality(code_blocks, language)
            
            # Track metrics
            self.code_generated_lines += sum(len(block['code'].split('\n')) for block in code_blocks)
            self.functions_created += sum(self._count_functions(block['code'], language) for block in code_blocks)
            if include_tests:
                self.tests_written += self._count_tests(response.content)
            
            return {
                "code": response.content,
                "confidence": response.confidence,
                "code_blocks": code_blocks,
                "quality_metrics": quality_metrics,
                "language": language,
                "includes_tests": include_tests,
                "includes_docs": include_docs,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Code generation error: {e}")
            raise
    
    def _extract_code_blocks(self, response: str) -> List[Dict[str, Any]]:
        """Extract code blocks from response"""
        code_blocks = []
        
        # Find code blocks with language specification
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for i, (language, code) in enumerate(matches):
            code_blocks.append({
                'id': i,
                'language': language or 'unknown',
                'code': code.strip(),
                'lines': len(code.strip().split('\n'))
            })
        
        return code_blocks
    
    async def _analyze_code_quality(self, code_blocks: List[Dict], language: str) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        if not code_blocks:
            return {}
        
        total_lines = sum(block['lines'] for block in code_blocks)
        
        # Combine all code for analysis
        all_code = '\n'.join(block['code'] for block in code_blocks)
        
        quality_metrics = {
            'total_lines': total_lines,
            'code_blocks': len(code_blocks),
            'has_comments': '#' in all_code or '//' in all_code or '/*' in all_code,
            'has_docstrings': '"""' in all_code or "'''" in all_code,
            'has_error_handling': any(term in all_code.lower() for term in ['try', 'except', 'catch', 'error']),
            'has_type_hints': ':' in all_code and '->' in all_code if language == 'python' else False,
            'complexity_score': self._calculate_complexity(all_code),
            'maintainability_score': self._calculate_maintainability(all_code)
        }
        
        return quality_metrics
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity (simplified)"""
        # Count control structures
        control_structures = ['if', 'for', 'while', 'switch', 'case', 'try', 'catch']
        complexity = sum(code.lower().count(structure) for structure in control_structures)
        
        # Normalize by lines of code
        lines = len(code.split('\n'))
        return complexity / max(lines, 1)
    
    def _calculate_maintainability(self, code: str) -> float:
        """Calculate maintainability score (simplified)"""
        score = 0.5  # Base score
        
        # Boost for comments and documentation
        if '#' in code or '//' in code:
            score += 0.2
        if '"""' in code or "'''" in code:
            score += 0.2
        
        # Boost for proper structure
        if 'def ' in code or 'function' in code:
            score += 0.1
        
        # Reduce for very long lines
        long_lines = sum(1 for line in code.split('\n') if len(line) > 100)
        if long_lines > 0:
            score -= min(0.2, long_lines * 0.05)
        
        return min(1.0, max(0.0, score))
    
    def _count_functions(self, code: str, language: str) -> int:
        """Count functions in code"""
        if language.lower() == 'python':
            return code.count('def ')
        elif language.lower() in ['javascript', 'typescript']:
            return code.count('function ') + code.count('=>')
        elif language.lower() == 'java':
            return len(re.findall(r'(public|private|protected).*?\s+\w+\s*\(', code))
        else:
            # Generic function counting
            return code.count('function') + code.count('def ')
    
    def _count_tests(self, response: str) -> int:
        """Count test functions in response"""
        test_patterns = [
            r'def test_\w+',
            r'it\([\'"].*?[\'"]',
            r'test\([\'"].*?[\'"]',
            r'@Test'
        ]
        
        test_count = 0
        for pattern in test_patterns:
            test_count += len(re.findall(pattern, response))
        
        return test_count
    
    async def review_code(
        self,
        code: str,
        language: str = "python",
        focus_areas: List[str] = None,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive code review
        """
        try:
            focus_text = ", ".join(focus_areas) if focus_areas else "all aspects"
            
            review_prompt = f"""
Perform a comprehensive code review of the following {language} code, focusing on {focus_text}:

```{language}
{code}
```

Please analyze:
1. Code Quality and Style
2. Performance and Efficiency
3. Security Considerations
4. Error Handling and Edge Cases
5. Maintainability and Readability
6. Best Practices Compliance
7. Potential Bugs or Issues
8. Suggestions for Improvement

Provide specific, actionable feedback with examples where appropriate.
"""
            
            response = await self.run(review_prompt, user_id, code_review=True)
            
            # Extract specific feedback categories
            feedback_analysis = self._analyze_review_feedback(response.content)
            
            return {
                "review": response.content,
                "confidence": response.confidence,
                "feedback_analysis": feedback_analysis,
                "language": language,
                "focus_areas": focus_areas or ["general"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Code review error: {e}")
            raise
    
    def _analyze_review_feedback(self, review: str) -> Dict[str, Any]:
        """Analyze review feedback for categorization"""
        categories = {
            "security_issues": len(re.findall(r'security|vulnerability|injection|xss', review, re.IGNORECASE)),
            "performance_issues": len(re.findall(r'performance|slow|optimize|efficient', review, re.IGNORECASE)),
            "style_issues": len(re.findall(r'style|format|naming|convention', review, re.IGNORECASE)),
            "bug_risks": len(re.findall(r'bug|error|exception|null|undefined', review, re.IGNORECASE)),
            "maintainability": len(re.findall(r'maintain|readable|complex|refactor', review, re.IGNORECASE)),
            "positive_feedback": len(re.findall(r'good|excellent|well|proper|correct', review, re.IGNORECASE))
        }
        
        return categories
    
    async def optimize_code(
        self,
        code: str,
        optimization_goals: List[str],
        language: str = "python",
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Optimize code for specific goals
        """
        try:
            goals_text = ", ".join(optimization_goals)
            
            optimization_prompt = f"""
Optimize the following {language} code for: {goals_text}

Original code:
```{language}
{code}
```

Please provide:
1. Optimized version of the code
2. Explanation of optimizations made
3. Performance impact analysis
4. Trade-offs and considerations
5. Before/after comparison

Focus on maintaining correctness while improving the specified aspects.
"""
            
            response = await self.run(optimization_prompt, user_id, optimize_code=True)
            
            # Extract optimized code
            optimized_blocks = self._extract_code_blocks(response.content)
            
            self.optimizations_made += 1
            
            return {
                "optimization": response.content,
                "confidence": response.confidence,
                "optimized_code": optimized_blocks,
                "optimization_goals": optimization_goals,
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Code optimization error: {e}")
            raise
    
    def get_creator_metrics(self) -> Dict[str, Any]:
        """Get Creator-specific performance metrics"""
        base_status = self.get_status()
        
        creator_metrics = {
            "code_generated_lines": self.code_generated_lines,
            "functions_created": self.functions_created,
            "tests_written": self.tests_written,
            "bugs_fixed": self.bugs_fixed,
            "optimizations_made": self.optimizations_made,
            "supported_languages": len(self.supported_languages),
            "frameworks_supported": len(self.frameworks),
            "improvement_iterations": self.improvement_iterations,
            "average_code_quality": self._calculate_average_quality(),
            "specialization": "Self-Improving Code Generation"
        }
        
        base_status["creator_metrics"] = creator_metrics
        return base_status
    
    def _calculate_average_quality(self) -> float:
        """Calculate average code quality score"""
        if self.total_interactions == 0:
            return 0.0
        
        # Simple heuristic based on various factors
        quality_factors = [
            self.functions_created / max(self.total_interactions, 1),
            self.tests_written / max(self.functions_created, 1),
            self.optimizations_made / max(self.total_interactions, 1)
        ]
        
        return min(1.0, sum(quality_factors) / len(quality_factors))

# Global Creator agent instance
creator_agent = CreatorAgent()
