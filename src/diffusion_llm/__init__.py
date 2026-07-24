"""Classroom DiffusionLLM.

Importing this package registers the supported diffusion model classes with
Hugging Face AutoModel. Run ``python -m diffusion_llm --help`` for the CLI.
"""

from diffusion_llm.modeling import register_diffusion_models

__version__ = "0.2.0"

register_diffusion_models()

__all__ = ["__version__", "register_diffusion_models"]
