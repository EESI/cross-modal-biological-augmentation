# text_to_image_generator.py

import os
import time
import pandas as pd
from PIL import Image
from io import BytesIO
import torch
from tqdm import tqdm

# Import configuration
from config_t2i_generator import T2I_CFG, make_prompt_all_features

# ====================================================
# 1. Setup Environment & Paths
# ====================================================
BASE_DIR = T2I_CFG["base_dir"]
MODEL_TYPE = T2I_CFG["model_type"]
OUT_DIR_NAME = T2I_CFG["out_dirs"][MODEL_TYPE]
FULL_OUT_DIR = os.path.join(BASE_DIR, OUT_DIR_NAME)

print(f" Mode: {MODEL_TYPE}")
print(f" Output Directory: {FULL_OUT_DIR}")

# ====================================================
# 2. Model Initialization
# ====================================================
gemini_client = None
qwen_pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"

if MODEL_TYPE == "gemini":
    from google import genai
    from google.genai import types

    api_key = T2I_CFG["gemini"]["api_key"]
    # Fallback to environment variable if key contains placeholder
    if "YOUR_GEMINI" in api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(" API Key for Gemini is missing.")

    gemini_client = genai.Client(api_key=api_key)
    print(" Gemini Client Initialized.")

elif MODEL_TYPE == "qwen":
    from diffusers import DiffusionPipeline
    
    print(f" Loading Diffusion Pipeline: {T2I_CFG['qwen']['model_id']}...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    try:
        qwen_pipe = DiffusionPipeline.from_pretrained(
            T2I_CFG['qwen']['model_id'], 
            torch_dtype=dtype
        )
        qwen_pipe = qwen_pipe.to(device)
        print(" Diffusion Pipeline Loaded.")
    except Exception as e:
        print(f" Failed to load model: {e}")
        print("Tip: Ensure you have access to the model and 'huggingface-cli login' is done if required.")
        exit()

# ====================================================
# 3. Generation Functions
# ====================================================

def generate_gemini(prompt, save_path):
    """Generates image using Google Gemini API."""
    try:
        response = gemini_client.models.generate_content(
            model=T2I_CFG["gemini"]["model_name"],
            contents=[prompt],
            config=types.GenerateContentConfig(response_modalities=['Image'])
        )
        
        # Extract image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    img = Image.open(BytesIO(part.inline_data.data))
                    img.save(save_path)
                    return True
        return False
    except Exception as e:
        print(f"  [Gemini Error] {e}")
        return False

def generate_qwen(prompt, save_path):
    """Generates image using Local Diffusion Pipeline."""
    try:
        cfg = T2I_CFG["qwen"]
        negative_prompt = (
            "blurry, low resolution, distorted, text, labels, human-made objects, "
            "bad anatomy, oversaturated, unrealistic colors"
        )
        generator = torch.Generator(device=device).manual_seed(cfg["seed"])
        
        image = qwen_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=cfg["width"],
            height=cfg["height"],
            num_inference_steps=cfg["num_inference_steps"],
            guidance_scale=cfg["guidance_scale"], # true_cfg_scale or guidance_scale depending on model
            generator=generator,
        ).images[0]
        
        image.save(save_path)
        return True
    except Exception as e:
        print(f"  [Diffusers Error] {e}")
        return False

# ====================================================
# 4. Main Execution Loop
# ====================================================
def run_t2i_pipeline():
    splits = ["train", "val", "test"]
    
    for split in splits:
        csv_file = T2I_CFG["csv_files"][split]
        csv_path = os.path.join(BASE_DIR, csv_file)
        
        # Create split specific output folder
        save_dir = os.path.join(FULL_OUT_DIR, split)
        os.makedirs(save_dir, exist_ok=True)
        
        if not os.path.exists(csv_path):
            print(f" CSV not found: {csv_path}. Skipping {split}.")
            continue
            
        df = pd.read_csv(csv_path)
        ###for test only
        # df = df[:1]
        
        # Limit for testing
        if T2I_CFG["limit_per_split"]:
            df = df.head(T2I_CFG["limit_per_split"])
            
        print(f"\n=== Processing Split: {split} ({len(df)} items) ===")
        
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{MODEL_TYPE} Generating"):
            save_filename = f"{split}_{idx}.png"
            save_path = os.path.join(save_dir, save_filename)
            
            # Resume Logic
            if T2I_CFG["resume"] and os.path.exists(save_path):
                skip_count += 1
                continue
            
            # Generate Prompt
            prompt = make_prompt_all_features(row)
            
            # Call Model
            is_success = False
            if MODEL_TYPE == "gemini":
                is_success = generate_gemini(prompt, save_path)
                time.sleep(T2I_CFG["gemini"]["sleep_interval"]) # Rate limiting
            elif MODEL_TYPE == "qwen":
                is_success = generate_qwen(prompt, save_path)
            
            if is_success:
                success_count += 1
            else:
                fail_count += 1
                
        print(f"Finished {split}:  Success={success_count}, ⏭ Skipped={skip_count},  Failed={fail_count}")

if __name__ == "__main__":
    run_t2i_pipeline()