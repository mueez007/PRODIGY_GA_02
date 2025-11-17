#!/usr/bin/env python3
"""
Working Image Generator - Stable Diffusion
Task 02: Image Generation with Pre-trained Models
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
from pathlib import Path
import time
import os

print("🚀 Working Image Generator - Stable Diffusion")
print("💻 Optimized for M4 MacBook Pro")

# Set environment variable to avoid Flax issues
os.environ['DISABLE_FLAX'] = '1'

class WorkingImageGenerator:
    def __init__(self):
        # Use MPS (Apple Silicon) if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("📱 Using: Apple Silicon GPU (MPS)")
        else:
            self.device = "cpu"
            print("📱 Using: CPU")
        
        self.pipe = None
    
    def load_model(self):
        """Load Stable Diffusion model without Flax dependencies"""
        try:
            print("📥 Loading model...")
            
            # Use a reliable model
            model_id = "runwayml/stable-diffusion-v1-5"
            
            # Load with safe settings
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                use_safetensors=True,
                safety_checker=None,  # Disable for speed
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
                print("   Enabled attention slicing for memory efficiency")
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_images(self, prompt, num_images=1, output_dir="output_images"):
        """Generate images from text prompt"""
        if not self.pipe:
            print("❌ Model not loaded")
            return None
        
        print(f"🎨 Generating {num_images} image(s) for: '{prompt}'")
        start_time = time.time()
        
        try:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            images = []
            for i in range(num_images):
                print(f"   Generating image {i+1}/{num_images}...")
                
                # Generate with basic settings
                with torch.no_grad():
                    result = self.pipe(
                        prompt=prompt,
                        num_inference_steps=20,  # Faster generation
                        guidance_scale=7.5,
                        height=512,
                        width=512
                    )
                
                image = result.images[0]
                images.append(image)
                
                # Save each image
                clean_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_prompt = clean_prompt[:30].replace(' ', '_')
                filename = f"{output_dir}/{clean_prompt}_{i+1}.png"
                image.save(filename)
                print(f"💾 Saved: {filename}")
            
            generation_time = time.time() - start_time
            print(f"✅ Generated {len(images)} images in {generation_time:.1f}s")
            return images
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Generate images from text')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--num-images', type=int, default=1, help='Number of images (1-2)')
    parser.add_argument('--output-dir', type=str, default='generated_images', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = WorkingImageGenerator()
    
    # Load model
    if generator.load_model():
        # Generate images
        images = generator.generate_images(
            prompt=args.prompt,
            num_images=args.num_images,
            output_dir=args.output_dir
        )
        
        if images:
            print(f"\n🎉 Success! Generated {len(images)} image(s)")
            # Show first image
            try:
                images[0].show()
            except:
                print("💡 Check the output directory for your images!")
        else:
            print("❌ Failed to generate images")
    else:
        print("❌ Could not initialize generator")

if __name__ == "__main__":
    main()
