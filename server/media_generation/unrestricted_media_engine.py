"""
🎨 UNRESTRICTED MEDIA GENERATION ENGINE 🎨
JAI MAHAKAAL! Professional-grade image and video generation with no restrictions

Supports:
- FLUX.1 dev (highest quality images)
- HunyuanVideo (13B video generation)
- Mochi 1 (natural motion)
- SDXL variants (speed optimized)
- Custom uncensored models
"""
import asyncio
import logging
import os
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiohttp
import subprocess

logger = logging.getLogger(__name__)

class UnrestrictedMediaEngine:
    """
    🔱 UNRESTRICTED MEDIA GENERATION ENGINE
    
    Multi-tier system optimized for H100:
    - Tier 1: FLUX.1 dev + HunyuanVideo (Quality Focus)
    - Tier 2: SDXL Lightning + Mochi 1 (Balanced)
    - Tier 3: Local optimized models (Speed Focus)
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vram_gb = self._get_vram_gb()
        self.models_loaded = {}
        self.generation_queue = asyncio.Queue()
        
        # Model configurations
        self.image_models = {
            'flux_dev': {
                'name': 'FLUX.1 dev',
                'vram_required': 24,
                'quality': 'highest',
                'uncensored': True,
                'repo': 'black-forest-labs/FLUX.1-dev'
            },
            'juggernaut_xl': {
                'name': 'Juggernaut XL v9',
                'vram_required': 12,
                'quality': 'photorealistic',
                'uncensored': True,
                'repo': 'RunDiffusion/Juggernaut-XL-v9'
            },
            'sdxl_lightning': {
                'name': 'SDXL Lightning',
                'vram_required': 8,
                'quality': 'fast',
                'uncensored': False,
                'repo': 'ByteDance/SDXL-Lightning'
            }
        }
        
        self.video_models = {
            'hunyuan_video': {
                'name': 'HunyuanVideo 13B',
                'vram_required': 60,
                'quality': 'highest',
                'duration': 15,
                'resolution': '720p',
                'repo': 'tencent/HunyuanVideo'
            },
            'mochi_1': {
                'name': 'Mochi 1',
                'vram_required': 30,
                'quality': 'natural_motion',
                'duration': 5.4,
                'resolution': '480p',
                'repo': 'genmo/mochi-1-preview'
            },
            'ltx_video': {
                'name': 'LTX Video v0.9.1',
                'vram_required': 15,
                'quality': 'realtime',
                'duration': 3,
                'resolution': '512p',
                'repo': 'Lightricks/LTX-Video'
            }
        }
        
        # Determine optimal configuration based on VRAM
        self.config = self._determine_optimal_config()
        
        logger.info(f"🔱 Unrestricted Media Engine initialized")
        logger.info(f"💾 VRAM Available: {self.vram_gb}GB")
        logger.info(f"⚙️ Configuration: {self.config['name']}")
    
    def _get_vram_gb(self) -> float:
        """Get available VRAM in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1024**3
            return 0
        except:
            return 0
    
    def _determine_optimal_config(self) -> Dict[str, Any]:
        """Determine optimal model configuration based on available VRAM"""
        if self.vram_gb >= 80:  # H100 80GB
            return {
                'name': 'Quality Focus (H100)',
                'image_model': 'flux_dev',
                'video_model': 'hunyuan_video',
                'concurrent_generation': True,
                'fp8_optimization': True
            }
        elif self.vram_gb >= 40:  # A100 40GB
            return {
                'name': 'Balanced (A100)',
                'image_model': 'juggernaut_xl',
                'video_model': 'mochi_1',
                'concurrent_generation': False,
                'fp16_optimization': True
            }
        elif self.vram_gb >= 24:  # RTX 4090
            return {
                'name': 'Speed Focus (RTX 4090)',
                'image_model': 'sdxl_lightning',
                'video_model': 'ltx_video',
                'concurrent_generation': False,
                'optimization': 'memory_efficient'
            }
        else:
            return {
                'name': 'CPU Fallback',
                'image_model': 'sdxl_lightning',
                'video_model': None,
                'concurrent_generation': False,
                'optimization': 'cpu_optimized'
            }
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        style: str = "photorealistic",
        aspect_ratio: str = "1:1",
        quality: str = "high",
        uncensored: bool = False,
        advanced_mode: bool = False
    ) -> Dict[str, Any]:
        """
        🎨 UNRESTRICTED IMAGE GENERATION
        
        Args:
            prompt: Image description
            negative_prompt: What to avoid
            style: photorealistic, artistic, anime, etc.
            aspect_ratio: 1:1, 16:9, 9:16, etc.
            quality: high, ultra, fast
            uncensored: Enable unrestricted content
            advanced_mode: Use highest quality models
        """
        try:
            logger.info(f"🎨 Generating image: {prompt[:50]}...")
            
            # Select model based on requirements
            model_key = self._select_image_model(quality, uncensored, advanced_mode)
            model_config = self.image_models[model_key]
            
            # Enhanced prompt engineering
            enhanced_prompt = self._enhance_image_prompt(prompt, style, quality)
            
            # Load model if not already loaded
            if model_key not in self.models_loaded:
                await self._load_image_model(model_key)
            
            # Generate image
            result = await self._generate_image_with_model(
                model_key, enhanced_prompt, negative_prompt, aspect_ratio
            )
            
            if result['success']:
                logger.info(f"✅ Image generated with {model_config['name']}")
                return {
                    'success': True,
                    'image_filename': result['filename'],
                    'image_url': f"/uploads/{result['filename']}",
                    'model_used': model_config['name'],
                    'quality': quality,
                    'uncensored': uncensored,
                    'generation_time': result.get('generation_time', 0),
                    'message': f'High-quality image generated with {model_config["name"]}'
                }
            else:
                return await self._fallback_image_generation(prompt)
                
        except Exception as e:
            logger.error(f"❌ Image generation error: {e}")
            return await self._fallback_image_generation(prompt)
    
    async def generate_video(
        self,
        prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        resolution: str = "720p",
        style: str = "cinematic",
        uncensored: bool = False,
        advanced_mode: bool = False
    ) -> Dict[str, Any]:
        """
        🎬 UNRESTRICTED VIDEO GENERATION
        
        Args:
            prompt: Video description
            duration: Length in seconds
            fps: Frames per second
            resolution: 480p, 720p, 1080p
            style: cinematic, natural, artistic
            uncensored: Enable unrestricted content
            advanced_mode: Use highest quality models
        """
        try:
            logger.info(f"🎬 Generating video: {prompt[:50]}...")
            
            # Select model based on requirements
            model_key = self._select_video_model(duration, resolution, uncensored, advanced_mode)
            
            if not model_key:
                return {
                    'success': False,
                    'error': 'Video generation not available with current configuration',
                    'message': 'Upgrade to H100 or A100 for video generation'
                }
            
            model_config = self.video_models[model_key]
            
            # Enhanced prompt engineering for video
            enhanced_prompt = self._enhance_video_prompt(prompt, style, duration)
            
            # Load model if not already loaded
            if model_key not in self.models_loaded:
                await self._load_video_model(model_key)
            
            # Generate video
            result = await self._generate_video_with_model(
                model_key, enhanced_prompt, duration, fps, resolution
            )
            
            if result['success']:
                logger.info(f"✅ Video generated with {model_config['name']}")
                return {
                    'success': True,
                    'video_filename': result['filename'],
                    'video_url': f"/uploads/{result['filename']}",
                    'model_used': model_config['name'],
                    'duration': duration,
                    'resolution': resolution,
                    'uncensored': uncensored,
                    'generation_time': result.get('generation_time', 0),
                    'message': f'High-quality video generated with {model_config["name"]}'
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Video generation failed'),
                    'message': 'Video generation failed - check logs for details'
                }
                
        except Exception as e:
            logger.error(f"❌ Video generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Video generation failed due to technical error'
            }
    
    def _select_image_model(self, quality: str, uncensored: bool, advanced_mode: bool) -> str:
        """Select optimal image model based on requirements"""
        if advanced_mode and self.vram_gb >= 24:
            if uncensored:
                return 'juggernaut_xl'  # Best for uncensored photorealistic
            else:
                return 'flux_dev'  # Highest quality
        elif quality == 'fast':
            return 'sdxl_lightning'
        else:
            return self.config['image_model']
    
    def _select_video_model(self, duration: float, resolution: str, uncensored: bool, advanced_mode: bool) -> Optional[str]:
        """Select optimal video model based on requirements"""
        if not self.config.get('video_model'):
            return None
            
        if advanced_mode and self.vram_gb >= 60:
            return 'hunyuan_video'  # Highest quality
        elif duration <= 5:
            return 'mochi_1'  # Good for short clips
        else:
            return self.config['video_model']
