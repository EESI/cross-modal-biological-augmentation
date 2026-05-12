# config_t2i_generator.py

import os

# ====================================================
# General Configuration
# ====================================================
T2I_CFG = {
    # --- Execution Mode ---
    "model_type": "gemini",   # Options: "gemini" (API) or "qwen" (Local GPU)
    
    # --- Paths ---
    # Base directory containing metadata CSV files
    "base_dir": "/path/to/dataset/",
    
    # Output directory names (relative to base_dir)
    "out_dirs": {
        "gemini": "images_gemini",
        "qwen":   "images_qwen"
    },

    # --- Metadata Files ---
    "csv_files": {
        "train": "meta_train_envfeature.csv",
        "val":   "meta_val_envfeature.csv",
        "test":  "meta_test_envfeature.csv"
    },

    # --- Gemini Settings (API) ---
    "gemini": {
        "api_key": "YOUR_GEMINI_API_KEY",  # Or set via os.environ["GEMINI_API_KEY"]
        "model_name": "gemini-2.5-flash-image", # Check for latest model availability
        "sleep_interval": 1.5  # To prevent rate limiting
    },

    # --- Qwen/Diffusers Settings (Local GPU) ---
    "qwen": {
        # Ensure this Model ID is valid on HuggingFace.         
        "model_id": "Qwen/Qwen-Image", 
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "width": 1024,
        "height": 1024,
        "seed": 42
    },
    
    # --- Columns to use for Prompt ---
    "features": ['env_biome', 'env_material', 'sample_type', 'scientific_name', 'empo_3'],
    
    # --- Debugging / Resume ---
    "limit_per_split": None,   # Set to integer (e.g., 50) for testing, None for full run
    "resume": True             # If True, skip existing images
}

# ====================================================
# Prompt Engineering Function
# ====================================================
def make_prompt_all_features(row):
    """
    Constructs a detailed prompt based on row metadata.
    """
    return (
        f"A highly detailed, photo-realistic scene of a natural environment representing "
        f"the biome '{row['env_biome']}', showing the material '{row['env_material']}' "
        f"and sample type '{row['sample_type']}' in its natural context. "
        f"The setting reflects the scientific sample '{row['scientific_name']}' categorized under "
        f"EMPO3 '{row['empo_3']}'. "
        f"Focus on environmental textures, terrain, and lighting — "
        f"no human-made objects, no laboratory tools, and no text or labels visible."
    )