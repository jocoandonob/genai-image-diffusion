import torch
from diffusers import StableDiffusionPipeline
import gc
from typing import Callable, Optional

def check_gpu_availability():
    """Check if a CUDA-compatible GPU is available."""
    return torch.cuda.is_available()

class ProgressCallback:
    def __init__(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        self.progress_callback = progress_callback
        self.current_step = 0
        self.total_steps = 50  # Default number of steps in Stable Diffusion

    def __call__(self, step: int, timestep: int, latents: torch.FloatTensor):
        if self.progress_callback:
            self.current_step = step
            progress = int((step / self.total_steps) * 100)
            self.progress_callback(progress, self.total_steps)

def generate_image(prompt, device="cpu", model_id="runwayml/stable-diffusion-v1-5", width=512, height=512, progress_callback=None):
    """
    Generate an image using Stable Diffusion based on the prompt.
    
    Args:
        prompt (str): The text prompt to generate an image from
        device (str): The device to use, either "cpu" or "gpu"
        model_id (str): The model ID to use for generation
        width (int): The width of the generated image
        height (int): The height of the generated image
        progress_callback (callable): Optional callback function to track progress
        
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
    
    # Create progress callback
    callback = ProgressCallback(progress_callback)
    
    # Generate the image with specified dimensions and progress tracking
    image = pipe(
        prompt,
        width=width,
        height=height,
        callback=callback,
        callback_steps=1
    ).images[0]
    
    # Clean up after generation
    pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return image
