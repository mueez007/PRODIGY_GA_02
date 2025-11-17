mage Generation with Stable Diffusion
Generate high-quality images from text descriptions using state-of-the-art Stable Diffusion models on your M4 MacBook Pro.

🚀 Quick Start
Installation
# Create project folder
mkdir task02_image_generation
cd task02_image_generation

# Create virtual environment (recommended)
python3 -m venv sd_env
source sd_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install diffusers accelerate transformers Pillow numpy safetensors


# Generate Your First Image
python3 working_generator.py --prompt "a cute robot painting a picture"

# Advanced Gen Options
python3 working_generator.py --prompt "sunset over mountains" --num-images 2 --output-dir my_images

# run demo
python3 working_demo.py

Options
--prompt: Text description (required)

--num-images: Number of images (1-3 recommended)

--output-dir: Output directory (default: generated_images)

🎨 Example Prompts
Animals & Characters:

"a cute cat wearing sunglasses, cartoon style"

"an astronaut riding a horse, photorealistic"

"a dragon reading a book, fantasy art"

Landscapes:

"sunset over a tranquil lake with mountains"

"magical forest with glowing mushrooms"

"beach with turquoise water and palm trees"

Art Styles:

"watercolor painting of autumn leaves"

"cyberpunk city street at night, neon lights"

"oil painting of flowers, van gogh style"

Fantasy & Sci-fi:

"futuristic city with flying cars"

"space station orbiting a colorful planet"

"wizard's tower in mystical forest"

💡 Tips for Better Results
Be Specific: "a red sports car on a mountain road" vs "a car"

Add Style Words: "photorealistic", "digital art", "watercolor painting"

Include Details: "with dramatic lighting", "highly detailed", "cinematic"

Experiment: Try different combinations of words

🛠️ Technical Details
Model: Stable Diffusion v1.5

Optimized for M4 MacBook Pro with 16GB RAM

Uses Apple Silicon GPU (MPS) when available

Image Size: 512x512 pixels

Format: PNG

⚡ Performance
Single Image: 30-60 seconds

Memory Usage: Optimized for 16GB RAM

First Run: Slower due to model download (~4GB)

🆘 Troubleshooting
Out of Memory:

Reduce --num-images to 1

Close other applications

Restart script

Slow Generation:

Normal for first generation

Subsequent runs are faster

Model Loading Issues:

Check internet connection

Ensure 4GB+ free storage