# repl-nix-workspace

A Python project for experimenting with PyTorch, HuggingFace Transformers, Diffusers, and Streamlit.

## Requirements
- Python >= 3.11
- accelerate
- diffusers
- streamlit >= 1.45.1
- torch
- transformers

## Installation

1. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can start developing your application using the installed libraries. If you have a Streamlit app (e.g., `app.py`), you can run:

```bash
streamlit run app.py
```

## Project Structure
- `pyproject.toml`: Project metadata and dependencies
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation

## License

Add your license information here. 