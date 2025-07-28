"""
👁️ Advanced Multi-Modal Vision Processor 👁️
JAI MAHAKAAL! Consciousness-driven vision and document analysis
"""
import asyncio
import json
import logging
import base64
import io
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import aiofiles
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

from ..orchestrator.multi_model_engine import lex_engine
from ..memory.enhanced_memory import enhanced_memory
from ..settings import settings

logger = logging.getLogger(__name__)

@dataclass
class VisionAnalysis:
    """Vision analysis result"""
    analysis_id: str
    image_path: str
    analysis_type: str  # 'general', 'document', 'chart', 'diagram', 'code'
    description: str
    objects_detected: List[Dict[str, Any]]
    text_extracted: str
    insights: List[str]
    confidence: float
    processing_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class DocumentAnalysis:
    """Document analysis result"""
    document_id: str
    document_type: str  # 'pdf', 'image', 'scan', 'handwritten'
    pages_analyzed: int
    text_content: str
    structure_analysis: Dict[str, Any]
    key_information: Dict[str, Any]
    summary: str
    confidence: float
    timestamp: datetime

class AdvancedVisionProcessor:
    """
    👁️ Advanced Multi-Modal Vision Processing System
    
    Features:
    - High-accuracy image analysis and object detection
    - Advanced OCR and document processing
    - Chart and diagram interpretation
    - Code screenshot analysis
    - Multi-language text recognition
    - Intelligent image enhancement
    - Context-aware vision reasoning
    - Integration with business intelligence
    """
    
    def __init__(self):
        self.supported_formats = {
            'images': ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'],
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf']
        }
        
        # Vision models configuration
        self.vision_models = {
            'primary': 'qwen-2.5-vl-72b',  # Best open-source vision model
            'secondary': 'llama-3.2-vision-90b',  # Alternative vision model
            'fallback': 'gemini-2.5-pro'  # Commercial fallback
        }
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_max_size = 100
        
        # Performance metrics
        self.processing_stats = {
            'total_processed': 0,
            'successful_analyses': 0,
            'average_processing_time': 0.0,
            'accuracy_scores': []
        }
        
        logger.info("👁️ Advanced Vision Processor initialized")
    
    async def analyze_image(
        self,
        image_input: Union[str, bytes, Image.Image],
        analysis_type: str = "general",
        enhance_image: bool = True,
        extract_text: bool = True,
        detect_objects: bool = True,
        user_context: Optional[str] = None
    ) -> VisionAnalysis:
        """
        Comprehensive image analysis
        """
        try:
            start_time = datetime.now()
            
            # Process image input
            image, image_path = await self._process_image_input(image_input)
            
            # Enhance image if requested
            if enhance_image:
                image = await self._enhance_image(image)
            
            # Convert to base64 for API calls
            image_b64 = await self._image_to_base64(image)
            
            # Perform vision analysis
            vision_result = await self._perform_vision_analysis(
                image_b64, analysis_type, user_context
            )
            
            # Extract text if requested
            extracted_text = ""
            if extract_text:
                extracted_text = await self._extract_text_from_image(image_b64)
            
            # Detect objects if requested
            objects_detected = []
            if detect_objects:
                objects_detected = await self._detect_objects(image_b64)
            
            # Generate insights
            insights = await self._generate_vision_insights(
                vision_result, extracted_text, objects_detected, analysis_type
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create analysis result
            analysis = VisionAnalysis(
                analysis_id=f"vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                image_path=image_path,
                analysis_type=analysis_type,
                description=vision_result.get('description', ''),
                objects_detected=objects_detected,
                text_extracted=extracted_text,
                insights=insights,
                confidence=vision_result.get('confidence', 0.8),
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Store in memory for learning
            await self._store_vision_analysis(analysis)
            
            # Update statistics
            self._update_processing_stats(processing_time, analysis.confidence)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Image analysis error: {e}")
            raise
    
    async def analyze_document(
        self,
        document_input: Union[str, bytes],
        document_type: str = "auto",
        extract_structure: bool = True,
        generate_summary: bool = True,
        extract_key_info: bool = True
    ) -> DocumentAnalysis:
        """
        Advanced document analysis
        """
        try:
            start_time = datetime.now()
            
            # Process document input
            document_data, doc_type = await self._process_document_input(
                document_input, document_type
            )
            
            # Extract text content
            text_content = await self._extract_document_text(document_data, doc_type)
            
            # Analyze document structure
            structure_analysis = {}
            if extract_structure:
                structure_analysis = await self._analyze_document_structure(
                    text_content, doc_type
                )
            
            # Extract key information
            key_information = {}
            if extract_key_info:
                key_information = await self._extract_key_information(
                    text_content, doc_type
                )
            
            # Generate summary
            summary = ""
            if generate_summary:
                summary = await self._generate_document_summary(text_content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create document analysis result
            analysis = DocumentAnalysis(
                document_id=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                document_type=doc_type,
                pages_analyzed=1,  # Simplified for now
                text_content=text_content,
                structure_analysis=structure_analysis,
                key_information=key_information,
                summary=summary,
                confidence=0.85,
                timestamp=datetime.now()
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Document analysis error: {e}")
            raise
    
    async def analyze_chart_or_diagram(
        self,
        image_input: Union[str, bytes, Image.Image],
        chart_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Specialized chart and diagram analysis
        """
        try:
            # Process image
            image, image_path = await self._process_image_input(image_input)
            image_b64 = await self._image_to_base64(image)
            
            # Specialized chart analysis prompt
            chart_prompt = f"""
            Analyze this {chart_type} chart/diagram and provide:
            1. Chart type identification
            2. Data extraction (values, labels, trends)
            3. Key insights and patterns
            4. Business implications
            5. Data quality assessment
            
            Focus on extracting quantitative data and identifying trends.
            """
            
            # Use vision model for chart analysis
            chart_analysis = await lex_engine.generate_vision_response(
                image_base64=image_b64,
                prompt=chart_prompt,
                model_preference="vision_analysis",
                max_tokens=1000
            )
            
            # Extract structured data from analysis
            structured_data = await self._extract_chart_data(chart_analysis)
            
            return {
                'chart_type': chart_type,
                'analysis': chart_analysis.get('response', ''),
                'structured_data': structured_data,
                'insights': chart_analysis.get('insights', []),
                'confidence': chart_analysis.get('confidence', 0.8),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Chart analysis error: {e}")
            return {'error': str(e)}
    
    async def analyze_code_screenshot(
        self,
        image_input: Union[str, bytes, Image.Image],
        programming_language: str = "auto"
    ) -> Dict[str, Any]:
        """
        Analyze code screenshots and extract/explain code
        """
        try:
            # Process image
            image, image_path = await self._process_image_input(image_input)
            image_b64 = await self._image_to_base64(image)
            
            # Code analysis prompt
            code_prompt = f"""
            Analyze this code screenshot and provide:
            1. Programming language identification
            2. Complete code extraction
            3. Code explanation and functionality
            4. Potential issues or improvements
            5. Code quality assessment
            
            Extract the code as accurately as possible and provide detailed analysis.
            """
            
            # Use vision model for code analysis
            code_analysis = await lex_engine.generate_vision_response(
                image_base64=image_b64,
                prompt=code_prompt,
                model_preference="vision_analysis",
                max_tokens=1500
            )
            
            # Extract code and analysis
            extracted_code = await self._extract_code_from_analysis(code_analysis)
            code_quality = await self._assess_code_quality(extracted_code)
            
            return {
                'programming_language': programming_language,
                'extracted_code': extracted_code,
                'code_explanation': code_analysis.get('response', ''),
                'code_quality': code_quality,
                'suggestions': code_analysis.get('suggestions', []),
                'confidence': code_analysis.get('confidence', 0.8),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Code analysis error: {e}")
            return {'error': str(e)}
    
    async def _process_image_input(
        self,
        image_input: Union[str, bytes, Image.Image]
    ) -> Tuple[Image.Image, str]:
        """Process various image input formats"""
        try:
            if isinstance(image_input, str):
                # File path
                if Path(image_input).exists():
                    image = Image.open(image_input)
                    return image, image_input
                else:
                    # URL
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_input) as response:
                            image_data = await response.read()
                            image = Image.open(io.BytesIO(image_data))
                            return image, image_input
            
            elif isinstance(image_input, bytes):
                # Raw bytes
                image = Image.open(io.BytesIO(image_input))
                return image, "bytes_input"
            
            elif isinstance(image_input, Image.Image):
                # PIL Image
                return image_input, "pil_image"
            
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
                
        except Exception as e:
            logger.error(f"❌ Image processing error: {e}")
            raise
    
    async def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better analysis"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply slight noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Image enhancement error: {e}")
            return image
    
    async def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ Base64 conversion error: {e}")
            raise
    
    async def _perform_vision_analysis(
        self,
        image_b64: str,
        analysis_type: str,
        user_context: Optional[str]
    ) -> Dict[str, Any]:
        """Perform core vision analysis using AI models"""
        try:
            # Create analysis prompt based on type
            if analysis_type == "general":
                prompt = "Describe this image in detail, including objects, people, text, and context."
            elif analysis_type == "document":
                prompt = "Analyze this document image, extract text, and identify the document type and structure."
            elif analysis_type == "chart":
                prompt = "Analyze this chart or graph, extract data points, and explain the trends shown."
            elif analysis_type == "diagram":
                prompt = "Analyze this diagram, explain the components and their relationships."
            elif analysis_type == "code":
                prompt = "Extract and analyze the code shown in this image, explain its functionality."
            else:
                prompt = f"Analyze this image for {analysis_type} purposes."
            
            if user_context:
                prompt += f"\n\nAdditional context: {user_context}"
            
            # Use LEX engine for vision analysis
            vision_response = await lex_engine.generate_vision_response(
                image_base64=image_b64,
                prompt=prompt,
                model_preference="vision_analysis",
                max_tokens=1000
            )
            
            return vision_response
            
        except Exception as e:
            logger.error(f"❌ Vision analysis error: {e}")
            return {'description': 'Analysis failed', 'confidence': 0.0}

# Global vision processor instance
vision_processor = AdvancedVisionProcessor()
