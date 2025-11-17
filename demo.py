#!/usr/bin/env python3
"""
Demo Script for Task 02 - Image Generation with Stable Diffusion
Generates multiple example images to showcase the model's capabilities
"""

from working_generator import WorkingImageGenerator
import time

def main():
    print("🤖 Task 02 Demo - Image Generation with Stable Diffusion")
    print("=" * 60)
    print("This demo will generate 4 different images to showcase various styles.")
    print("Each image will take 30-60 seconds to generate.")
    print("-" * 60)
    
    # Initialize generator
    generator = WorkingImageGenerator()
    
    # Load model
    print("📥 Loading Stable Diffusion model...")
    if not generator.load_model():
        print("❌ Failed to load model. Exiting demo.")
        return
    
    # Example prompts showcasing different styles
    prompts = [
        "a cute robot painting a picture, digital art, vibrant colors",
        "sunset over a mountain lake, photorealistic, highly detailed", 
        "watercolor painting of autumn leaves, artistic, soft colors",
        "futuristic cyberpunk city at night, neon lights, detailed buildings"
    ]
    
    print(f"\n🎨 Generating 1 image for each of {len(prompts)} prompts...")
    print("This will take 2-4 minutes total...")
    print("-" * 60)
    
    total_start = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Generating: '{prompt}'")
        print("⏳ This may take 30-60 seconds...")
        
        # Generate 1 image per prompt
        images = generator.generate_images(
            prompt=prompt,
            num_images=1,
            output_dir="demo_output"
        )
        
        if images:
            print(f"✅ Completed prompt {i}/{len(prompts)}")
        else:
            print(f"❌ Failed for prompt: {prompt}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🎉 Demo completed in {total_time:.1f} seconds!")
    print("📁 Check the 'demo_output' folder for your generated images!")
    print("✨ You can now create your own images using:")
    print("   python3 working_generator.py --prompt 'your description here'")
    print("=" * 60)

if __name__ == "__main__":
    main()