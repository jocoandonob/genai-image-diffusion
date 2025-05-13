import torch
from diffusers import StableDiffusionPipeline
import gc

def check_gpu_availability():
    """Check if a CUDA-compatible GPU is available."""
    return torch.cuda.is_available()

def generate_image(prompt, device="cpu", model_id="runwayml/stable-diffusion-v1-5", width=512, height=512):
    """
    Generate an image using Stable Diffusion based on the prompt.
    
    Args:
        prompt (str): The text prompt to generate an image from
        device (str): The device to use, either "cpu" or "gpu"
        model_id (str): The model ID to use for generation
        width (int): The width of the generated image
        height (int): The height of the generated image
        
    Returns:
        PIL.Image: The generated image
    """
    # Clean up any existing models and free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Determine the appropriate device
    if device == "gpu" and torch.cuda.is_available():
        torch_device = "cuda"
    else:
        torch_device = "cpu"
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    
    # Move to the appropriate device
    pipe = pipe.to(torch_device)
    
    # Disable safety checker to prevent black images
    pipe.safety_checker = None
    
    # Generate the image with specified dimensions
    image = pipe(prompt, width=width, height=height).images[0]
    
    # Clean up after generation
    pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return image
