import streamlit as st
import time
import random
import io
from PIL import Image, ImageDraw
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Stable Diffusion Image Generator with JOCO",
    page_icon="🎨",
    layout="wide"
)

# App title and description
st.title("Stable Diffusion Image Generator with JOCO")
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
    
    # Calculate estimated time based on device
    estimated_time = 60 if device == "CPU" else 15  # Base time in seconds
    
    # Display estimated time
    st.info(f"Estimated generation time: {estimated_time} seconds")
    
    # Generation button
    generate_button = st.button("Generate Image", type="primary", use_container_width=True)
    
    st.markdown("### Additional Information")
    st.markdown("""
    - Generation time varies based on device
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
        # Add a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress, total_steps):
            progress_bar.progress(progress)
            status_text.text(f"Generating image... {progress}% complete")

        # Generate the image
        with st.spinner(f"Generating image... This may take {estimated_time} seconds"):
            try:
                # Generate the image with progress tracking
                image = generate_image(
                    prompt=prompt,
                    device=device,
                    width=512,
                    height=512,
                    progress_callback=update_progress
                )
                
                # Display the generated image
                st.image(image, caption="Generated Image", width=512)
                
                # Add download button
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                st.download_button(
                    label="Download Image",
                    data=img_byte_arr,
                    file_name="generated_image_512x512.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your dependencies and try again.")

# Display initial instructions if no image has been generated
if 'generate_button' not in locals() or not generate_button:
    st.markdown("""
    ## How to use:
    1. Enter a descriptive prompt in the sidebar
    2. Choose your processing device (if available)
    3. Select a Stable Diffusion model
    4. Click "Generate Image"
    5. Wait for the generation to complete (estimated time shown in sidebar)
    6. Download your image if desired
    
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
