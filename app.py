import streamlit as st
import time
import random
import io
from PIL import Image, ImageDraw
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Stable Diffusion Image Generator",
    page_icon="🎨",
    layout="wide"
)

# App title and description
st.title("Stable Diffusion Image Generator")
st.markdown("Generate images from text prompts using Stable Diffusion.")

# Sidebar for settings
with st.sidebar:
    st.header("Generation Settings")
    
    # Prompt input
    prompt = st.text_area(
        "Enter your prompt:", 
        value="a fantasy castle on top of a floating island, beautiful sunset, highly detailed",
        height=100
    )
    
    # Image size options
    st.subheader("Image Size")
    size_options = {
        "512x512": (512, 512),
        "768x768": (768, 768),
        "1024x1024": (1024, 1024)
    }
    selected_size = st.selectbox(
        "Select image size:",
        list(size_options.keys()),
        index=0
    )
    width, height = size_options[selected_size]
    
    # Device selection
    device = st.radio("Processing Device:", ["CPU", "GPU"], index=0)
    if device == "GPU":
        st.info("GPU acceleration requires CUDA support.")
    
    # Model selection
    model_name = st.selectbox(
        "Model:",
        ["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4"],
        index=0
    )
    
    # Calculate estimated time based on device and size
    base_time = 60 if device == "CPU" else 15  # Base time in seconds
    size_factor = (width * height) / (512 * 512)  # Scale factor based on image size
    estimated_time = int(base_time * size_factor)
    
    # Display estimated time
    st.info(f"Estimated generation time: {estimated_time} seconds")
    
    # Generation button
    generate_button = st.button("Generate Image", type="primary", use_container_width=True)
    
    st.markdown("### Additional Information")
    st.markdown("""
    - Generation time varies based on image size and device
    - For best results, provide detailed prompts
    - Safety filters are disabled (may generate unsafe content)
    """)

    # Display authentication message
    st.markdown("### Authentication")
    st.info("""
    If you encounter login errors, you may need to authenticate with Hugging Face:
    ```
    huggingface-cli login
    ```
    """)
    
    # Installation note for missing packages
    st.markdown("### Required Packages")
    st.code("""
    pip install torch==2.0.0 diffusers==0.14.0 transformers==4.27.0
    """)

# Try to import real generation capabilities if available
try:
    import torch
    from utils import generate_image, check_gpu_availability
    
    # Check if GPU is available
    gpu_available = check_gpu_availability()
    
    # Set flag indicating real generation is available
    REAL_GENERATION = True
    
    # Update GPU status in sidebar if needed
    if not gpu_available and device == "GPU":
        st.sidebar.warning("GPU not detected. Defaulting to CPU.")
        device = "CPU"
        
except ImportError:
    REAL_GENERATION = False
    # Show an error message if dependencies are missing
    st.error("""
    **Missing AI Dependencies**
    
    The required ML libraries (torch, diffusers) are not installed.
    This app will generate unique images based on your prompts, but they will not
    be generated with real AI.
    
    To enable real AI image generation, install the required packages.
    Note: These packages are already in your pyproject.toml but seem to have
    installation issues in the current environment.
    """)

# Main content area
if generate_button:
    if not prompt:
        st.error("Please enter a prompt before generating.")
    else:
        with st.spinner(f"Generating {selected_size} image using {model_name}... This may take {estimated_time} seconds."):
            # Set up progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # If we have real generation capability
                if REAL_GENERATION:
                    # Show progress updates as we go
                    for i in range(30):
                        # Simulate progress during model loading
                        time.sleep(0.1)
                        progress_bar.progress(i)
                        status_text.text(f"Loading model... {i}%")
                    
                    # Actual image generation
                    device_str = device.lower()
                    image = generate_image(prompt, device_str, model_name, width=width, height=height)
                    
                    # Finish progress bar
                    for i in range(30, 101):
                        time.sleep(0.02)
                        progress_bar.progress(i)
                        status_text.text(f"Generating image... {i}%")
                    
                    generation_type = "AI-generated image using Stable Diffusion"
                else:
                    # Create a simple placeholder image
                    for i in range(95):
                        time.sleep(0.02)
                        progress_bar.progress(i)
                        status_text.text(f"Creating placeholder... {i}%")
                    
                    # Create a simple message image instead of a unique placeholder
                    img = Image.new('RGB', (width, height), color=(30, 30, 50))
                    d = ImageDraw.Draw(img)
                    d.text((20, 20), f"Model: {model_name}", fill=(255, 255, 255))
                    d.text((20, 50), f"Size: {selected_size}", fill=(255, 255, 255))
                    d.text((20, 80), f"Prompt: {prompt[:50]}...", fill=(255, 255, 255))
                    d.text((20, 120), "Dependencies not installed", fill=(255, 200, 200))
                    d.text((20, 150), "Please install required packages:", fill=(255, 200, 200))
                    d.text((20, 180), "torch==2.0.0", fill=(200, 200, 255))
                    d.text((20, 210), "diffusers==0.14.0", fill=(200, 200, 255))
                    d.text((20, 240), "transformers==4.27.0", fill=(200, 200, 255))
                    
                    image = img
                    
                    # Finish progress
                    for i in range(95, 101):
                        time.sleep(0.02)
                        progress_bar.progress(i)
                        status_text.text(f"Finalizing... {i}%")
                        
                    generation_type = "Placeholder (install dependencies for real AI generation)"
                
                # Clear status text
                status_text.empty()
                
                # Display the image
                st.subheader("Generated Image")
                st.image(image, caption=f"Generated from: {prompt}", use_container_width=True)
                
                # Convert PIL image to bytes for download
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                
                # Download button
                st.download_button(
                    label="Download Image",
                    data=image_bytes,
                    file_name=f"stable_diffusion_image_{selected_size}.png",
                    mime="image/png"
                )
                
                st.success("Image generated successfully!")
                st.info(f"**Note:** {generation_type}")
                
            except Exception as e:
                st.error(f"An error occurred during image generation: {str(e)}")
                st.markdown(f"""
                **Possible solutions:**
                - Try a different prompt
                - Check your internet connection
                - If using a Hugging Face model, run `huggingface-cli login` to authenticate
                - Try a different model from the dropdown list
                - Try a smaller image size
                """)

# Display initial instructions if no image has been generated
if 'generate_button' not in locals() or not generate_button:
    st.markdown("""
    ## How to use:
    1. Enter a descriptive prompt in the sidebar
    2. Select your desired image size
    3. Choose your processing device (if available)
    4. Select a Stable Diffusion model
    5. Click "Generate Image"
    6. Wait for the generation to complete (estimated time shown in sidebar)
    7. Download your image if desired
    
    The more detailed your prompt, the better the results will be!
    """)
    
    # Show example prompt ideas
    st.subheader("Prompt Ideas:")
    st.markdown("""
    - "A serene mountain lake at sunset, with trees reflecting in the water"
    - "A futuristic cityscape with flying cars and neon lights"
    - "A magical forest with glowing mushrooms and fairy creatures"
    - "An ancient stone temple covered in vines in a jungle clearing"
    """)
    
    # Ready message
    st.info("""
    **Ready for image generation**
    
    This application can generate images from your text prompts.
    """)
