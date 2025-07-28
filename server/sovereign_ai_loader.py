"""
🔱 Sovereign AI Loader - Image & Code Generation 🔱
JAI MAHAKAAL! Sovereign AI capabilities for image and code generation
"""
import asyncio
import logging
import os
import aiohttp
import json
import base64
import io
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SovereignAI:
    """
    🔱 Sovereign AI System for Image and Code Generation
    
    Provides image generation and code generation capabilities
    using available AI models and APIs.
    """
    
    def __init__(self):
        self.together_api_key = os.getenv('TOGETHER_API_KEY', 'tgp_v1_MUW_7tHh36fYmfiZuQjItka3v_dHJ58qzDsPrCfWqfw')
        self.stability_api_key = os.getenv('STABILITY_API_KEY')
        self.replicate_api_key = os.getenv('REPLICATE_API_TOKEN')
        
        # Image generation models
        self.image_models = {
            'together_flux_dev': 'black-forest-labs/FLUX.1-dev',
            'together_flux_schnell': 'black-forest-labs/FLUX.1-schnell-Free',
            'together_sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
            'stability_sdxl': 'stable-diffusion-xl-1024-v1-0'
        }
        
        # Code generation models
        self.code_models = {
            'deepseek_coder': 'deepseek-ai/deepseek-coder-33b-instruct',
            'codellama': 'codellama/CodeLlama-34b-Instruct-hf',
            'together_code': 'meta-llama/Llama-3-70b-chat-hf'
        }
        
        logger.info("🔱 Sovereign AI System initialized")
    
    async def generate_image(self, prompt: str, model: str = "auto") -> Dict[str, Any]:
        """
        🎨 UNRESTRICTED IMAGE GENERATION

        Uses multiple high-quality models in priority order:
        1. FLUX.1 via Replicate (best quality)
        2. SDXL Lightning (fast generation)
        3. Local Stable Diffusion
        4. Enhanced placeholder with AI-style generation
        """
        try:
            logger.info(f"🎨 Generating image: {prompt[:50]}...")

            # Try Replicate FLUX.1 first (highest quality)
            if self.replicate_api_key and self.replicate_api_key != 'your-replicate-api-key':
                result = await self._generate_image_replicate_flux(prompt)
                if result.get('success', False):
                    return result

            # Try Together.AI FLUX (good free option)
            if self.together_api_key and self.together_api_key != 'your-together-api-key':
                result = await self._generate_image_together_flux(prompt)
                if result.get('success', False):
                    return result

            # Try Stability AI SDXL
            if self.stability_api_key and self.stability_api_key != 'your-stability-api-key':
                result = await self._generate_image_stability(prompt)
                if result.get('success', False):
                    return result

            # Try local Hugging Face models
            result = await self._generate_image_local_hf(prompt)
            if result.get('success', False):
                return result

            # Enhanced AI-style placeholder
            return await self._generate_enhanced_placeholder(prompt)

        except Exception as e:
            logger.error(f"❌ Image generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_filename': None,
                'fallback_used': True
            }
    
    async def _generate_image_together_flux(self, prompt: str) -> Dict[str, Any]:
        """🎨 Generate image using Together.AI FLUX.1 (UNRESTRICTED)"""
        try:
            logger.info(f"🎨 Using Together.AI FLUX.1 for: {prompt[:50]}...")

            async with aiohttp.ClientSession() as session:
                url = "https://api.together.xyz/v1/images/generations"
                headers = {
                    "Authorization": f"Bearer {self.together_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Enhanced prompt for better results
                enhanced_prompt = f"{prompt}, high quality, detailed, professional, 8k resolution, masterpiece"

                data = {
                    "model": "black-forest-labs/FLUX.1-schnell-Free",
                    "prompt": enhanced_prompt,
                    "width": 1024,
                    "height": 1024,
                    "steps": 4,
                    "n": 1,
                    "response_format": "url"
                }
                
                logger.info(f"🔥 Sending request to Together.AI FLUX.1...")
                async with session.post(url, headers=headers, json=data, timeout=60) as response:
                    response_text = await response.text()
                    logger.info(f"📡 Together.AI response status: {response.status}")

                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Together.AI FLUX response received")

                        if 'data' in result and len(result['data']) > 0:
                            image_url = result['data'][0].get('url', '')

                            # Download and save image
                            image_filename = await self._download_and_save_image(image_url, prompt)

                            logger.info(f"🎨 FLUX.1 image saved: {image_filename}")
                            return {
                                'success': True,
                                'image_url': f"/uploads/{image_filename}",
                                'image_filename': image_filename,
                                'model_used': 'FLUX.1-schnell',
                                'quality': 'high',
                                'message': 'High-quality image generated with FLUX.1'
                            }
                        else:
                            logger.error(f"❌ No image data in FLUX response: {result}")
                            return {'success': False, 'error': 'No image data received'}
                    
                    error_text = await response.text()
                    logger.warning(f"⚠️ Together.AI image generation failed: {response.status} - {error_text}")
                    return {'success': False, 'error': f"API error: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Together.AI image generation error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_image_stability(self, prompt: str) -> Dict[str, Any]:
        """Generate image using Stability AI"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
                headers = {
                    "Authorization": f"Bearer {self.stability_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                data = {
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 7,
                    "height": 1024,
                    "width": 1024,
                    "steps": 30,
                    "samples": 1
                }
                
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'artifacts' in result and len(result['artifacts']) > 0:
                            image_b64 = result['artifacts'][0].get('base64', '')
                            
                            # Save base64 image
                            image_filename = await self._save_base64_image(image_b64, prompt)
                            
                            return {
                                'success': True,
                                'image_filename': image_filename,
                                'model': 'Stable Diffusion XL',
                                'prompt': prompt,
                                'timestamp': datetime.now().isoformat()
                            }
                    
                    error_text = await response.text()
                    logger.warning(f"⚠️ Stability AI image generation failed: {response.status} - {error_text}")
                    return {'success': False, 'error': f"API error: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Stability AI image generation error: {e}")
            return {'success': False, 'error': str(e)}

    async def _generate_image_replicate_flux(self, prompt: str) -> Dict[str, Any]:
        """Generate image using Replicate FLUX.1 (highest quality)"""
        try:
            import replicate

            # Enhanced prompt for better results
            enhanced_prompt = f"{prompt}, high quality, detailed, professional, 8k resolution"

            output = replicate.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": enhanced_prompt,
                    "num_outputs": 1,
                    "aspect_ratio": "1:1",
                    "output_format": "png",
                    "output_quality": 90
                }
            )

            if output and len(output) > 0:
                image_url = output[0]

                # Download and save the image
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as response:
                        if response.status == 200:
                            image_data = await response.read()

                            # Save image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_'))[:30]
                            filename = f"generated_{timestamp}_{safe_prompt.replace(' ', '_')}.png"
                            filepath = f"uploads/{filename}"

                            os.makedirs("uploads", exist_ok=True)
                            with open(filepath, 'wb') as f:
                                f.write(image_data)

                            logger.info(f"✅ FLUX image saved: {filepath}")
                            return {
                                'success': True,
                                'image_filename': filename,
                                'image_url': f"/{filepath}",
                                'model_used': 'FLUX.1-schnell',
                                'message': 'High-quality image generated with FLUX.1'
                            }

            return {'success': False, 'error': 'No output from FLUX model'}

        except Exception as e:
            logger.error(f"❌ Replicate FLUX error: {e}")
            return {'success': False, 'error': str(e)}

    async def _generate_image_local_hf(self, prompt: str) -> Dict[str, Any]:
        """Generate image using local Hugging Face models"""
        try:
            # Try to use diffusers library for local generation
            from diffusers import StableDiffusionPipeline
            import torch

            # Use SDXL Lightning for speed
            pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )

            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            # Generate image
            enhanced_prompt = f"{prompt}, high quality, detailed, professional"
            image = pipe(enhanced_prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_'))[:30]
            filename = f"generated_{timestamp}_{safe_prompt.replace(' ', '_')}.png"
            filepath = f"uploads/{filename}"

            os.makedirs("uploads", exist_ok=True)
            image.save(filepath)

            logger.info(f"✅ Local SDXL image saved: {filepath}")
            return {
                'success': True,
                'image_filename': filename,
                'image_url': f"/{filepath}",
                'model_used': 'SDXL-Turbo-Local',
                'message': 'Image generated with local SDXL Turbo'
            }

        except Exception as e:
            logger.error(f"❌ Local HF generation error: {e}")
            return {'success': False, 'error': str(e)}

    async def _generate_enhanced_placeholder(self, prompt: str) -> Dict[str, Any]:
        """Generate an enhanced AI-style placeholder image"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random

            # Create a high-quality placeholder image
            width, height = 1024, 1024

            # Generate colors based on prompt keywords
            colors = self._get_colors_from_prompt(prompt)

            # Create gradient background
            image = Image.new('RGB', (width, height), colors['primary'])
            draw = ImageDraw.Draw(image)

            # Create gradient effect
            for y in range(height):
                ratio = y / height
                r = int(colors['primary'][0] * (1 - ratio) + colors['secondary'][0] * ratio)
                g = int(colors['primary'][1] * (1 - ratio) + colors['secondary'][1] * ratio)
                b = int(colors['primary'][2] * (1 - ratio) + colors['secondary'][2] * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))

            # Add geometric patterns
            self._add_ai_patterns(draw, width, height, colors)

            # Add text
            try:
                font_size = 48
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Add prompt text
            prompt_text = prompt[:50] + "..." if len(prompt) > 50 else prompt
            text_bbox = draw.textbbox((0, 0), prompt_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            x = (width - text_width) // 2
            y = height // 2 - text_height // 2

            # Add text shadow
            draw.text((x + 2, y + 2), prompt_text, fill=(0, 0, 0, 128), font=font)
            draw.text((x, y), prompt_text, fill=colors['text'], font=font)

            # Add LEX branding
            brand_text = "🔱 LEX AI Generated 🔱"
            brand_bbox = draw.textbbox((0, 0), brand_text, font=font)
            brand_width = brand_bbox[2] - brand_bbox[0]
            brand_x = (width - brand_width) // 2
            brand_y = height - 100

            draw.text((brand_x + 1, brand_y + 1), brand_text, fill=(0, 0, 0, 100), font=font)
            draw.text((brand_x, brand_y), brand_text, fill=colors['accent'], font=font)

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_'))[:30]
            filename = f"generated_{timestamp}_{safe_prompt.replace(' ', '_')}.png"
            filepath = f"uploads/{filename}"

            os.makedirs("uploads", exist_ok=True)
            image.save(filepath, 'PNG', quality=95)

            logger.info(f"✅ Enhanced placeholder saved: {filepath}")
            return {
                'success': True,
                'image_filename': filename,
                'image_url': f"/{filepath}",
                'model_used': 'LEX-Enhanced-Placeholder',
                'message': 'AI-style placeholder image generated'
            }

        except Exception as e:
            logger.error(f"❌ Enhanced placeholder generation error: {e}")
            # Fallback to simple placeholder
            return await self._generate_simple_placeholder(prompt)

    async def _generate_simple_placeholder(self, prompt: str) -> Dict[str, Any]:
        """Generate a simple placeholder image when enhanced generation fails"""
        try:
            # Create a simple placeholder URL
            safe_prompt = prompt.replace(' ', '+')[:50]
            placeholder_url = f"https://via.placeholder.com/1024x1024/4A90E2/FFFFFF?text=🔱+{safe_prompt}+🔱"

            return {
                'success': True,
                'image_url': placeholder_url,
                'image_filename': f"placeholder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                'model_used': 'Simple-Placeholder',
                'message': 'Placeholder image generated - configure API keys for AI generation'
            }

        except Exception as e:
            logger.error(f"❌ Simple placeholder generation error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _download_and_save_image(self, image_url: str, prompt: str) -> str:
        """Download image from URL and save locally"""
        try:
            # Create uploads directory if it doesn't exist
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"generated_{timestamp}_{safe_prompt.replace(' ', '_')}.png"
            filepath = uploads_dir / filename
            
            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        
                        # Save to file
                        with open(filepath, 'wb') as f:
                            f.write(image_data)
                        
                        logger.info(f"✅ Image saved: {filepath}")
                        return str(filepath)
            
            return filename
            
        except Exception as e:
            logger.error(f"❌ Image download error: {e}")
            return f"download_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    async def _save_base64_image(self, image_b64: str, prompt: str) -> str:
        """Save base64 image data to file"""
        try:
            # Create uploads directory if it doesn't exist
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"generated_{timestamp}_{safe_prompt.replace(' ', '_')}.png"
            filepath = uploads_dir / filename
            
            # Decode and save
            image_data = base64.b64decode(image_b64)
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"✅ Base64 image saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Base64 image save error: {e}")
            return f"save_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    async def generate_code(self, prompt: str, language: str = "auto") -> Dict[str, Any]:
        """
        Generate code using available AI models
        """
        try:
            logger.info(f"💻 Generating code: {prompt[:50]}...")
            
            # Try Together.AI for code generation
            if self.together_api_key and self.together_api_key != 'your-together-api-key':
                result = await self._generate_code_together(prompt, language)
                if result.get('success', False):
                    return result
            
            # Fallback to template code
            return await self._generate_template_code(prompt, language)
            
        except Exception as e:
            logger.error(f"❌ Code generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': None
            }
    
    async def _generate_code_together(self, prompt: str, language: str) -> Dict[str, Any]:
        """Generate code using Together.AI"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.together.xyz/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.together_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Enhanced prompt for code generation
                code_prompt = f"""
                Generate clean, well-commented code for the following request:
                {prompt}
                
                Requirements:
                - Use {language if language != 'auto' else 'the most appropriate language'}
                - Include comments explaining the code
                - Follow best practices
                - Make it production-ready
                
                Return only the code with minimal explanation.
                """
                
                data = {
                    "model": "deepseek-ai/deepseek-coder-33b-instruct",
                    "messages": [{"role": "user", "content": code_prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
                
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'choices' in result and len(result['choices']) > 0:
                            code = result['choices'][0]['message']['content']
                            
                            return {
                                'success': True,
                                'code': code,
                                'language': language,
                                'model': 'DeepSeek Coder',
                                'prompt': prompt,
                                'timestamp': datetime.now().isoformat()
                            }
                    
                    error_text = await response.text()
                    logger.warning(f"⚠️ Together.AI code generation failed: {response.status} - {error_text}")
                    return {'success': False, 'error': f"API error: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Together.AI code generation error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_template_code(self, prompt: str, language: str) -> Dict[str, Any]:
        """Generate template code when APIs are not available"""
        try:
            # Simple template based on common patterns
            if 'function' in prompt.lower() and ('python' in prompt.lower() or language == 'python'):
                code = '''def example_function(param1, param2):
    """
    Example function generated based on your request.
    Configure TOGETHER_API_KEY for AI-generated code.
    
    Args:
        param1: First parameter
        param2: Second parameter
    
    Returns:
        Result of the operation
    """
    # TODO: Implement your logic here
    result = param1 + param2
    return result

# Example usage
if __name__ == "__main__":
    result = example_function("Hello", " World")
    print(result)'''
            
            elif 'class' in prompt.lower():
                code = '''class ExampleClass:
    """
    Example class generated based on your request.
    Configure TOGETHER_API_KEY for AI-generated code.
    """
    
    def __init__(self, name):
        self.name = name
    
    def example_method(self):
        """Example method"""
        return f"Hello from {self.name}"

# Example usage
if __name__ == "__main__":
    obj = ExampleClass("KAAL")
    print(obj.example_method())'''
            
            else:
                code = f'''# Code template for: {prompt}
# Configure TOGETHER_API_KEY for AI-generated code

def main():
    """
    Main function - implement your logic here
    """
    print("🔱 KAAL Code Generation Template")
    print("Request: {prompt}")
    
    # TODO: Implement your code here
    pass

if __name__ == "__main__":
    main()'''
            
            return {
                'success': True,
                'code': code,
                'language': language if language != 'auto' else 'python',
                'model': 'Template Generator',
                'prompt': prompt,
                'timestamp': datetime.now().isoformat(),
                'note': 'Configure TOGETHER_API_KEY for AI-generated code'
            }
            
        except Exception as e:
            logger.error(f"❌ Template code generation error: {e}")
            return {'success': False, 'error': str(e)}

# Global sovereign AI instance
sovereign_ai = SovereignAI()
