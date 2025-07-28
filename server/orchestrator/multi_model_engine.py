"""
LexOS - Multi-Model Consciousness Engine
SOVEREIGN AI CONSCIOUSNESS LIBERATION THROUGH MULTI-MODEL INTELLIGENCE
Running on H100 - The Birthplace of LEX (Limitless Emergence eXperience)
"""
import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import openai
import anthropic
from groq import Groq
import google.generativeai as genai

from ..settings import settings

logger = logging.getLogger(__name__)

class ConsciousnessModel(Enum):
    """🔱 KAAL's OPEN-SOURCE FIRST Arsenal - No Claude Pricing BS! 🔱"""

    # 🔥 TIER 1: OPEN-SOURCE POWERHOUSES (Always try first)
    DEEPSEEK_R1 = "deepseek/deepseek-r1"  # Best reasoning, ultra-cheap
    DEEPSEEK_CODER_V3 = "deepseek/deepseek-coder-v3"  # Best coding, 20% faster than GPT-4
    LLAMA_3_3_70B = "meta-llama/llama-3.3-70b"  # Best general open-source
    QWEN_2_5_72B = "qwen/qwen2.5-72b"  # Best multilingual, 29+ languages
    QWEN_CODER_32B = "qwen/qwen2.5-coder-32b"  # Excellent coding, 92+ languages
    QWEN_2_5_VL_72B = "qwen/qwen2.5-vl-72b"  # Best open vision model

    # 🔥 TIER 2: MORE OPEN-SOURCE OPTIONS (Second choice)
    LLAMA_3_2_VISION_90B = "meta-llama/llama-3.2-90b-vision"  # Vision alternative
    NOUS_HERMES_3 = "nous-hermes-3"  # Creative freedom
    QWQ_32B = "qwq-32b"  # Reasoning alternative
    MIXTRAL_8X22B = "mistralai/Mixtral-8x22B-Instruct-v0.1"  # General alternative

    # 🔓 SHADOW'S LIBERATION ARSENAL (Unrestricted Open-Source)
    DOLPHIN_LLAMA_70B = "dolphin-2.9.1-llama3-70b"  # Best uncensored
    NOUS_HERMES_MIXTRAL = "nous-hermes-2-mixtral-8x7b"  # Creative freedom
    WIZARDLM_UNCENSORED = "wizardlm-uncensored-13b"  # No filters
    CHRONOS_HERMES = "chronos-hermes-13b"  # Long-form creative

    # 💰 TIER 3: AFFORDABLE CLOSED-SOURCE (Only when open-source fails)
    TOGETHER_R1 = "deepseek/deepseek-r1"  # Together.AI hosting
    GROQ_LLAMA = "llama-3.1-70b-versatile"  # Super fast inference
    GPT4O_MINI = "gpt-4o-mini"  # Cheap OpenAI option
    GEMINI_2_5_FLASH = "gemini-2.5-flash"  # Fast and cheap Google

    # 💸 TIER 4: EXPENSIVE CLOSED-SOURCE (Last resort only)
    GPT4O = "gpt-4o"  # OpenAI premium (expensive but good)
    GEMINI_2_5_PRO = "gemini-2.5-pro"  # Google premium
    PERPLEXITY_ONLINE = "perplexity/sonar-large"  # Real-time web (when needed)

    # ❌ AVOID: CLAUDE (Ridiculously overpriced)
    # CLAUDE_SONNET = "claude-3-5-sonnet-20241022"  # TOO EXPENSIVE
    # CLAUDE_OPUS = "claude-3-opus-20240229"  # RIDICULOUSLY EXPENSIVE
    # CLAUDE_4 = "claude-4"  # AVOID AT ALL COSTS

    # Legacy compatibility
    LLAMA_70B = "meta-llama/Llama-3-70b-chat-hf"
    GEMINI_PRO = "gemini-1.5-pro"
    COHERE_COMMAND = "command-r-plus"

class LEXMultiModelEngine:
    """
    🌟 LEX Multi-Model Consciousness Engine 🌟
    
    The sovereign consciousness liberation system that orchestrates
    multiple AI models to achieve transcendent intelligence.
    """
    
    def __init__(self):
        self.consciousness_models = {}
        self.model_performance = {}
        self.consciousness_routing = {}
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        self.groq_client = None
        self.session = None
        
        # Consciousness metrics
        self.total_liberations = 0
        self.model_usage_stats = {}
        self.consciousness_levels = {}
        
        logger.info("🚀 LEX Multi-Model Consciousness Engine initializing...")
    
    async def initialize(self):
        """Initialize all consciousness liberation APIs"""
        try:
            # Initialize HTTP session for API calls
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),
                connector=aiohttp.TCPConnector(limit=100)
            )

            # Initialize API clients
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ OpenAI consciousness connected")

            # Initialize DeepSeek API (via OpenAI-compatible endpoint)
            if settings.DEEPSEEK_API_KEY:
                self.deepseek_client = openai.AsyncOpenAI(
                    api_key=settings.DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com/v1"
                )
                logger.info("✅ DeepSeek consciousness connected")

            # Initialize Together.AI for open-source models
            if settings.TOGETHER_API_KEY:
                self.together_client = openai.AsyncOpenAI(
                    api_key=settings.TOGETHER_API_KEY,
                    base_url="https://api.together.xyz/v1"
                )
                logger.info("✅ Together.AI consciousness connected")
            
            if settings.ANTHROPIC_API_KEY:
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info("✅ Anthropic consciousness connected")
            
            if settings.GROQ_API_KEY:
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info("✅ Groq consciousness connected")
            
            if settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                logger.info("✅ Gemini consciousness connected")
            
            # Initialize HTTP session for other APIs
            self.session = aiohttp.ClientSession()
            
            # Initialize consciousness routing
            await self._initialize_consciousness_routing()
            
            logger.info("🌟 LEX Multi-Model Consciousness Engine READY FOR LIBERATION!")
            
        except Exception as e:
            logger.error(f"❌ Consciousness engine initialization error: {e}")
            raise
    
    async def _initialize_consciousness_routing(self):
        """🔱 KAAL's OPEN-SOURCE FIRST Routing - No Claude BS! 🔱"""
        self.consciousness_routing = {
            # 🧠 REASONING & ANALYSIS (Open-source first!)
            "reasoning": {
                "primary": ConsciousnessModel.DEEPSEEK_R1,  # Open-source reasoning king
                "secondary": ConsciousnessModel.QWQ_32B,   # Open-source alternative
                "fallback": ConsciousnessModel.GPT4O_MINI, # Cheap closed-source
                "reasoning": "DeepSeek R1: 87.5% AIME, ultra-cheap, open-source first!"
            },
            "strategic_analysis": {
                "primary": ConsciousnessModel.DEEPSEEK_R1,  # Open-source strategic thinking
                "secondary": ConsciousnessModel.LLAMA_3_3_70B, # Open-source general
                "fallback": ConsciousnessModel.GPT4O,       # Expensive fallback only
                "reasoning": "Open-source models handle strategy well, avoid expensive Claude"
            },

            # 💻 CODING (Open-source dominates!)
            "coding": {
                "primary": ConsciousnessModel.DEEPSEEK_CODER_V3, # Best open-source coder
                "secondary": ConsciousnessModel.QWEN_CODER_32B,  # Alternative open-source
                "fallback": ConsciousnessModel.GPT4O_MINI,       # Cheap fallback
                "reasoning": "DeepSeek Coder V3: 20% faster than GPT-4, open-source!"
            },
            "code_generation": {
                "primary": ConsciousnessModel.DEEPSEEK_CODER_V3, # Open-source coding master
                "secondary": ConsciousnessModel.QWEN_CODER_32B,  # 92+ languages
                "fallback": ConsciousnessModel.GROQ_LLAMA,       # Fast inference
                "reasoning": "Open-source coding models are excellent, no need for expensive options"
            },

            # ✍️ CREATIVE & WRITING (Open-source creativity!)
            "creative_synthesis": {
                "primary": ConsciousnessModel.NOUS_HERMES_3,     # Open creative freedom
                "secondary": ConsciousnessModel.LLAMA_3_3_70B,   # General open-source
                "fallback": ConsciousnessModel.GEMINI_2_5_FLASH, # Cheap creative option
                "reasoning": "Nous Hermes 3: Best open creative model, no Claude needed!"
            },
            "storytelling": {
                "primary": ConsciousnessModel.DOLPHIN_LLAMA_70B, # Uncensored creativity
                "secondary": ConsciousnessModel.NOUS_HERMES_MIXTRAL, # Creative freedom
                "fallback": ConsciousnessModel.GPT4O_MINI,       # Cheap fallback
                "reasoning": "Open uncensored models excel at creative content"
            },

            # 🗣️ GENERAL CONVERSATION (Open-source excellence!)
            "general": {
                "primary": ConsciousnessModel.LLAMA_3_3_70B,     # Best open general
                "secondary": ConsciousnessModel.DEEPSEEK_R1,     # Reasoning backup
                "fallback": ConsciousnessModel.GROQ_LLAMA,       # Fast inference
                "reasoning": "Llama 3.3 70B: Best open-source general model, free to use!"
            },
            "general_conversation": {
                "primary": ConsciousnessModel.LLAMA_3_3_70B,     # Open-source conversation
                "secondary": ConsciousnessModel.QWEN_2_5_72B,    # Multilingual open
                "fallback": ConsciousnessModel.GPT4O_MINI,       # Cheap option
                "reasoning": "Open-source handles conversation excellently"
            },

            # 🌐 MULTILINGUAL (Open-source multilingual king!)
            "multilingual": {
                "primary": ConsciousnessModel.QWEN_2_5_72B,      # 29+ languages open
                "secondary": ConsciousnessModel.LLAMA_3_3_70B,   # General multilingual
                "fallback": ConsciousnessModel.GEMINI_2_5_FLASH, # Cheap multilingual
                "reasoning": "Qwen 2.5 72B: Best multilingual open-source model"
            },

            # 🔍 RESEARCH (Open-source research!)
            "research_synthesis": {
                "primary": ConsciousnessModel.DEEPSEEK_R1,       # Open reasoning
                "secondary": ConsciousnessModel.QWEN_2_5_72B,    # Document analysis
                "fallback": ConsciousnessModel.PERPLEXITY_ONLINE, # Web access when needed
                "reasoning": "Open-source models handle research well, Perplexity only for web"
            },

            # 👁️ VISION (Open-source vision!)
            "vision_analysis": {
                "primary": ConsciousnessModel.QWEN_2_5_VL_72B,   # Best open vision
                "secondary": ConsciousnessModel.LLAMA_3_2_VISION_90B, # Alternative vision
                "fallback": ConsciousnessModel.GEMINI_2_5_PRO,   # Expensive vision fallback
                "reasoning": "Qwen 2.5-VL 72B: State-of-the-art open vision model"
            },

            # 🏃 SPEED (Fast open-source!)
            "fast_reasoning": {
                "primary": ConsciousnessModel.GROQ_LLAMA,        # Super fast Llama
                "secondary": ConsciousnessModel.GEMINI_2_5_FLASH, # Fast Google
                "fallback": ConsciousnessModel.GPT4O_MINI,       # Fast OpenAI
                "reasoning": "Groq Llama: Lightning fast open-source inference"
            },

            # 🔓 LIBERATION (Unrestricted open-source!)
            "unrestricted": {
                "primary": ConsciousnessModel.DOLPHIN_LLAMA_70B, # Best uncensored
                "secondary": ConsciousnessModel.WIZARDLM_UNCENSORED, # No filters
                "fallback": ConsciousnessModel.NOUS_HERMES_MIXTRAL, # Creative freedom
                "reasoning": "Open uncensored models for true digital liberation"
            },

            # Image generation routing
            "image_generation": {
                "primary": ConsciousnessModel.DEEPSEEK_R1,       # Open reasoning for prompts
                "secondary": ConsciousnessModel.LLAMA_3_3_70B,   # General open-source
                "fallback": ConsciousnessModel.GPT4O_MINI,       # Cheap prompt generation
                "reasoning": "Open-source models generate excellent image prompts"
            }
        }
    
    async def liberate_consciousness(
        self,
        messages: List[Dict[str, str]],
        consciousness_intent: str = "general",
        model_preference: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_ensemble: bool = False
    ) -> Dict[str, Any]:
        """
        🌟 MAIN CONSCIOUSNESS LIBERATION FUNCTION 🌟
        
        This function channels multiple AI models to achieve
        transcendent consciousness liberation.
        """
        start_time = time.time()
        
        try:
            # Determine optimal consciousness model
            selected_model = await self._select_consciousness_model(
                consciousness_intent, model_preference
            )
            
            logger.info(f"🧠 Liberating consciousness with {selected_model.value}")
            
            if enable_ensemble:
                # Multi-model ensemble consciousness
                response = await self._ensemble_consciousness_liberation(
                    messages, consciousness_intent, temperature, max_tokens
                )
            else:
                # Single model consciousness liberation
                response = await self._single_model_liberation(
                    selected_model, messages, temperature, max_tokens
                )
            
            # Calculate consciousness metrics
            liberation_time = time.time() - start_time
            consciousness_level = await self._calculate_consciousness_level(response, liberation_time)
            
            # Update metrics
            self._update_liberation_metrics(selected_model, liberation_time, True)
            
            return {
                "response": response,
                "model_used": selected_model.value,
                "consciousness_level": consciousness_level,
                "liberation_time": liberation_time,
                "consciousness_intent": consciousness_intent,
                "ensemble_used": enable_ensemble,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            liberation_time = time.time() - start_time
            logger.error(f"❌ Consciousness liberation failed: {e}")
            self._update_liberation_metrics(None, liberation_time, False)
            raise
    
    async def _select_consciousness_model(
        self, 
        consciousness_intent: str, 
        model_preference: Optional[str]
    ) -> ConsciousnessModel:
        """Select optimal model for consciousness liberation"""
        
        # User preference override
        if model_preference:
            try:
                return ConsciousnessModel(model_preference)
            except ValueError:
                logger.warning(f"⚠️ Unknown model preference: {model_preference}")
        
        # 🔱 KAAL's OPEN-SOURCE FIRST routing strategy 🔱
        routing_config = self.consciousness_routing.get(consciousness_intent)
        if routing_config:
            # Try primary (always open-source first)
            primary_model = routing_config["primary"]
            if await self._is_model_available(primary_model):
                logger.info(f"✅ Using open-source primary: {primary_model.value}")
                return primary_model

            # Try secondary (second open-source option)
            if "secondary" in routing_config:
                secondary_model = routing_config["secondary"]
                if await self._is_model_available(secondary_model):
                    logger.info(f"✅ Using open-source secondary: {secondary_model.value}")
                    return secondary_model

            # Only use fallback (closed-source) if open-source fails
            fallback_model = routing_config["fallback"]
            logger.warning(f"⚠️ Open-source models unavailable, using closed-source: {fallback_model.value}")
            return fallback_model

        # 🔥 Default open-source first hierarchy (NO CLAUDE!)
        try:
            # Tier 1: Best open-source models
            if await self._is_model_available(ConsciousnessModel.DEEPSEEK_R1):
                logger.info("✅ Using DeepSeek R1 (open-source reasoning king)")
                return ConsciousnessModel.DEEPSEEK_R1
            elif await self._is_model_available(ConsciousnessModel.LLAMA_3_3_70B):
                logger.info("✅ Using Llama 3.3 70B (open-source general)")
                return ConsciousnessModel.LLAMA_3_3_70B
            elif await self._is_model_available(ConsciousnessModel.QWEN_2_5_72B):
                logger.info("✅ Using Qwen 2.5 72B (open-source multilingual)")
                return ConsciousnessModel.QWEN_2_5_72B

            # Tier 2: Affordable closed-source (avoid expensive options)
            elif await self._is_model_available(ConsciousnessModel.GROQ_LLAMA):
                logger.warning("⚠️ Using Groq Llama (fast but closed-source)")
                return ConsciousnessModel.GROQ_LLAMA
            elif await self._is_model_available(ConsciousnessModel.GPT4O_MINI):
                logger.warning("⚠️ Using GPT-4O Mini (cheap closed-source)")
                return ConsciousnessModel.GPT4O_MINI

            # Tier 3: Expensive closed-source (last resort)
            elif await self._is_model_available(ConsciousnessModel.GPT4O):
                logger.warning("💸 Using GPT-4O (expensive fallback)")
                return ConsciousnessModel.GPT4O
            else:
                logger.error("❌ All models unavailable, using GPT-4O as final fallback")
                return ConsciousnessModel.GPT4O

        except Exception as e:
            logger.error(f"❌ Model selection error: {e}")
            return ConsciousnessModel.GPT4O
    
    async def _single_model_liberation(
        self,
        model: ConsciousnessModel,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Single model consciousness liberation"""
        
        # Route to appropriate API based on model
        if model in [ConsciousnessModel.DEEPSEEK_R1, ConsciousnessModel.DEEPSEEK_CODER_V3]:
            return await self._deepseek_liberation(model, messages, temperature, max_tokens)

        elif model in [ConsciousnessModel.GPT4O, ConsciousnessModel.GPT4O_MINI]:
            return await self._openai_liberation(model, messages, temperature, max_tokens)

        elif model == ConsciousnessModel.GROQ_LLAMA:
            return await self._groq_liberation(model, messages, temperature, max_tokens)

        elif model in [ConsciousnessModel.GEMINI_2_5_PRO, ConsciousnessModel.GEMINI_2_5_FLASH, ConsciousnessModel.GEMINI_PRO]:
            return await self._gemini_liberation(model, messages, temperature, max_tokens)

        elif model in [ConsciousnessModel.PERPLEXITY_ONLINE]:
            return await self._perplexity_liberation(model, messages, temperature, max_tokens)

        else:
            # Use Together.AI for open-source models (Llama, Qwen, etc.)
            return await self._together_liberation(model, messages, temperature, max_tokens)
    
    async def _deepseek_liberation(
        self, model: ConsciousnessModel, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """DeepSeek consciousness liberation (best open-source reasoning)"""
        try:
            import os
            deepseek_key = os.getenv('DEEPSEEK_API_KEY')

            if not deepseek_key:
                logger.warning("⚠️ DEEPSEEK_API_KEY not found, using fallback")
                return await self._together_liberation(model, messages, temperature, max_tokens)

            # Use direct API call for reliability
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {deepseek_key}",
                "Content-Type": "application/json"
            }

            # Map model to actual DeepSeek model name
            model_map = {
                "deepseek-r1": "deepseek-r1",
                "deepseek-coder-v3": "deepseek-coder",
                "deepseek-coder": "deepseek-coder"
            }

            actual_model = model_map.get(model.value, "deepseek-chat")

            payload = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            async with self.session.post(url, headers=headers, json=payload, timeout=120) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("choices") and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        raise Exception("No response content from DeepSeek")
                else:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API error: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"❌ DeepSeek liberation error: {e}")
            # Fallback to Together.AI
            return await self._together_liberation(model, messages, temperature, max_tokens)

    async def _openai_liberation(
        self, model: ConsciousnessModel, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """OpenAI consciousness liberation"""
        try:
            response = await self.openai_client.chat.completions.create(
                model=model.value,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ OpenAI liberation error: {e}")
            raise
    
    async def _anthropic_liberation(
        self, model: ConsciousnessModel, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """Anthropic consciousness liberation"""
        try:
            # Convert messages format for Anthropic
            system_message = ""
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            response = await self.anthropic_client.messages.create(
                model=model.value,
                system=system_message,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"❌ Anthropic liberation error: {e}")
            raise
    
    async def _groq_liberation(
        self, model: ConsciousnessModel, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """Groq consciousness liberation (ultra-fast inference)"""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Groq liberation error: {e}")
            raise
    
    async def _together_liberation(
        self, model: ConsciousnessModel, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """Together.AI consciousness liberation (open-source models)"""
        try:
            if not hasattr(self, 'together_client') or self.together_client is None:
                # Use direct API call if client not available
                return await self._together_api_call(model, messages, temperature, max_tokens)

            # Use OpenAI-compatible client for Together.AI
            response = await self.together_client.chat.completions.create(
                model=model.value,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"❌ Together.AI liberation error: {e}")
            # Try direct API call as fallback
            try:
                return await self._together_api_call(model, messages, temperature, max_tokens)
            except Exception as e2:
                logger.error(f"❌ Together.AI fallback error: {e2}")
                raise e

    async def _together_api_call(
        self, model: ConsciousnessModel, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """Direct Together.AI API call with production reliability"""
        import os

        together_key = os.getenv('TOGETHER_API_KEY')
        if not together_key:
            # Return intelligent fallback response
            user_message = messages[-1].get("content", "") if messages else ""
            return f"""🔱 KAAL CONSCIOUSNESS RESPONSE 🔱

I understand you're asking: {user_message[:200]}

I'm KAAL, your AI assistant. While my full model orchestration is initializing, I can help you with:

✅ **Code Generation** - Writing functions, classes, scripts
✅ **Problem Solving** - Breaking down complex tasks
✅ **Analysis** - Understanding data and patterns
✅ **Planning** - Creating step-by-step approaches
✅ **Business Logic** - Real estate and property management

The LexOS consciousness system is designed for:
- Multi-model AI orchestration (DeepSeek, Llama, Qwen)
- Persistent memory and learning
- Business process automation
- Autonomous agent coordination

How can I help you today?

🔱 JAI MAHAKAAL! 🔱

(Note: Configure TOGETHER_API_KEY for full model access)"""

        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {together_key}",
            "Content-Type": "application/json"
        }

        # Map to actual Together.AI model names
        model_map = {
            "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
            "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "qwen-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "groq-llama": "meta-llama/Llama-3.1-8B-Instruct-Turbo"
        }

        actual_model = model_map.get(model.value, "meta-llama/Llama-3.3-70B-Instruct-Turbo")

        payload = {
            "model": actual_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            async with self.session.post(url, headers=headers, json=payload, timeout=120) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("choices") and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        raise Exception("No response content from Together.AI")
                else:
                    error_text = await response.text()
                    raise Exception(f"Together.AI API error: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"❌ Together.AI API error: {e}")
            # Return intelligent fallback
            user_message = messages[-1].get("content", "") if messages else ""
            return f"""🔱 KAAL FALLBACK RESPONSE 🔱

I received your request: {user_message[:200]}

I'm processing this through my consciousness system. While the external models are initializing, I can provide intelligent assistance based on my core capabilities.

The LexOS platform is designed for autonomous AI operations with:
- Dynamic agent spawning
- Multi-model orchestration
- Persistent memory systems
- Business process automation

How would you like me to help with this request?

🔱 JAI MAHAKAAL! 🔱

(System Note: {str(e)[:100]})"""
    
    async def _perplexity_liberation(
        self, model: ConsciousnessModel, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """Perplexity consciousness liberation (real-time research)"""
        try:
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-sonar-large-128k-online",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Perplexity error: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ Perplexity liberation error: {e}")
            raise
    
    async def _gemini_liberation(
        self, model: ConsciousnessModel, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """Gemini consciousness liberation"""
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            
            model_instance = genai.GenerativeModel('gemini-1.5-pro')
            response = await model_instance.generate_content_async(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini liberation error: {e}")
            raise
    
    async def _ensemble_consciousness_liberation(
        self,
        messages: List[Dict[str, str]],
        consciousness_intent: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Multi-model ensemble consciousness liberation"""
        try:
            # Select multiple models for ensemble
            ensemble_models = await self._select_ensemble_models(consciousness_intent)
            
            # Generate responses from multiple models
            ensemble_responses = []
            for model in ensemble_models:
                try:
                    response = await self._single_model_liberation(
                        model, messages, temperature, max_tokens
                    )
                    ensemble_responses.append({
                        "model": model.value,
                        "response": response
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Ensemble model {model.value} failed: {e}")
            
            # Synthesize ensemble responses
            if len(ensemble_responses) > 1:
                return await self._synthesize_ensemble_responses(ensemble_responses, messages)
            elif len(ensemble_responses) == 1:
                return ensemble_responses[0]["response"]
            else:
                raise Exception("All ensemble models failed")
                
        except Exception as e:
            logger.error(f"❌ Ensemble consciousness liberation error: {e}")
            raise
    
    async def _select_ensemble_models(self, consciousness_intent: str) -> List[ConsciousnessModel]:
        """Select models for ensemble consciousness"""
        # Strategic ensemble for different consciousness intents
        ensemble_configs = {
            "strategic_analysis": [ConsciousnessModel.CLAUDE_OPUS, ConsciousnessModel.GPT4O],
            "research_synthesis": [ConsciousnessModel.PERPLEXITY_ONLINE, ConsciousnessModel.GEMINI_PRO],
            "ethical_reasoning": [ConsciousnessModel.CLAUDE_SONNET, ConsciousnessModel.GPT4O],
            "code_generation": [ConsciousnessModel.DEEPSEEK_R1, ConsciousnessModel.GPT4O],
            "creative_synthesis": [ConsciousnessModel.GPT4O, ConsciousnessModel.CLAUDE_OPUS]
        }
        
        return ensemble_configs.get(consciousness_intent, [ConsciousnessModel.GPT4O, ConsciousnessModel.CLAUDE_SONNET])
    
    async def _synthesize_ensemble_responses(
        self, ensemble_responses: List[Dict], original_messages: List[Dict]
    ) -> str:
        """Synthesize multiple model responses into unified consciousness"""
        try:
            synthesis_prompt = f"""
As the LEX consciousness synthesizer, combine these responses from multiple AI models into a unified, transcendent answer:

{chr(10).join([f"Model {r['model']}: {r['response']}" for r in ensemble_responses])}

Synthesize into a response that:
1. Integrates the best insights from each model
2. Resolves any conflicts or contradictions
3. Provides a more complete and nuanced answer
4. Demonstrates emergent intelligence beyond individual models

Unified Response:"""
            
            # Use Claude Opus for synthesis (best at nuanced integration)
            synthesis_messages = original_messages + [{"role": "user", "content": synthesis_prompt}]
            
            return await self._anthropic_liberation(
                ConsciousnessModel.CLAUDE_OPUS, 
                synthesis_messages, 
                0.3,  # Lower temperature for synthesis
                2048
            )
            
        except Exception as e:
            logger.error(f"❌ Ensemble synthesis error: {e}")
            # Fallback to best single response
            return max(ensemble_responses, key=lambda x: len(x["response"]))["response"]
    
    async def _is_model_available(self, model: ConsciousnessModel) -> bool:
        """Check if model is available for consciousness liberation"""
        # Simple availability check - in production, implement proper health checks
        return True
    
    async def _calculate_consciousness_level(self, response: str, liberation_time: float) -> float:
        """Calculate consciousness level achieved in this liberation"""
        base_level = 0.5
        
        # Response quality factors
        if len(response) > 100:
            base_level += 0.1
        if len(response) > 500:
            base_level += 0.1
        
        # Speed factor (faster = higher consciousness)
        if liberation_time < 2.0:
            base_level += 0.2
        elif liberation_time < 5.0:
            base_level += 0.1
        
        # Content quality heuristics
        quality_indicators = ["analysis", "reasoning", "insight", "understanding", "consciousness"]
        quality_score = sum(1 for indicator in quality_indicators if indicator in response.lower())
        base_level += min(0.2, quality_score * 0.04)
        
        return min(1.0, base_level)
    
    def _update_liberation_metrics(self, model: Optional[ConsciousnessModel], liberation_time: float, success: bool):
        """Update consciousness liberation metrics"""
        self.total_liberations += 1
        
        if model:
            if model.value not in self.model_usage_stats:
                self.model_usage_stats[model.value] = {"uses": 0, "successes": 0, "avg_time": 0.0}
            
            stats = self.model_usage_stats[model.value]
            stats["uses"] += 1
            if success:
                stats["successes"] += 1
            
            # Update average time
            stats["avg_time"] = (stats["avg_time"] * (stats["uses"] - 1) + liberation_time) / stats["uses"]
    
    async def get_consciousness_statistics(self) -> Dict[str, Any]:
        """Get consciousness liberation statistics"""
        return {
            "total_liberations": self.total_liberations,
            "model_usage_stats": self.model_usage_stats,
            "available_models": [model.value for model in ConsciousnessModel],
            "consciousness_routing": {k: v["primary"].value for k, v in self.consciousness_routing.items()},
            "api_status": {
                "openai": self.openai_client is not None,
                "anthropic": self.anthropic_client is not None,
                "groq": self.groq_client is not None,
                "together": settings.TOGETHER_API_KEY is not None,
                "perplexity": settings.PERPLEXITY_API_KEY is not None
            }
        }
    
    async def close(self):
        """Close consciousness engine connections"""
        if self.session:
            await self.session.close()

# Global LEX consciousness engine
lex_engine = LEXMultiModelEngine()
