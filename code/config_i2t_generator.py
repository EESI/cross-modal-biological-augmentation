# config_generator.py
import os
import json

# ====================================================
# 1. Paths & Settings
# ====================================================
GEN_CFG = {
    # --- mode settings ---
    "model_type": "openai",    # "openai" (API) or "qwen" (Local GPU)
    "split_mode": "part1",     # "part0" (first 50%), "part1" (second 50%), "full" (full)
    
    # --- path settings ---
    "root_dir": "/path/to/dataset/",
    # "splits_dir": "splits", # root_dir/splits
    # "images_dir": "images", # root_dir/images
    
    # --- model settings ---
    "openai": {
        "api_key": "YOUR_OPENAI_API_KEY_HERE", # api key 
        "model_name": "gpt-4o-mini",
        "max_retries": 3,
        "sleep_base": 1.5
    },
    
    "qwen": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_new_tokens": 2400,
        "device_map": "auto"
    },

    # --- common param ---
    "checkpoint_every": 50,  # checkpoint for the number of photos
    "force_reprocess": False # rewrite or not
}

# ====================================================
# 2. JSON Schema (EHR structure)
# ====================================================
EHR_JSON_SCHEMA = {
    "name": "MiniEHR",
    "schema": {
        "type": "object",
        "properties": {
            "patient": {"type":"object","properties":{
                "age_years":{"type":"integer"},
                "sex":{"type":"string","enum":["male","female","unknown"]}},
                "required":["age_years","sex"]},
            "encounter":{"type":"object","properties":{
                "encounter_type":{"type":"string","enum":["dermatology_outpatient","unknown"]},
                "site":{"type":"string"}},
                "required":["encounter_type","site"]},
            "lesion_observation":{"type":"object","properties":{
                "anatomical_site":{"type":"string"},
                "visual_findings":{"type":"array","items":{"type":"string"}},
                "size_mm":{"type":"number"},
                "image_quality_note":{"type":"string"}},
                "required":["anatomical_site"]},
            "assessment":{"type":"object","properties":{
                "provisional_diagnosis_label":{"type":"string"},
                "malignancy_risk":{"type":"number","minimum":0,"maximum":1},
                "rationale":{"type":"array","items":{"type":"string"}}},
                "required":["provisional_diagnosis_label","malignancy_risk","rationale"]},
            "orders":{"type":"array","items":{"type":"object","properties":{
                "type":{"type":"string"}, "note":{"type":"string"}}}}
        },
        "required":["patient","encounter","lesion_observation","assessment"]
    }
}

# ====================================================
# 3. System Prompt
# ====================================================
SYSTEM_PROMPT = (
    "You are a clinical documentation assistant for dermatology research. "
    "Extract structured EHR-like data from the dermoscopic image and the given metadata. "
    "Return ONLY valid JSON that matches the provided JSON Schema (no prose). "
    "If uncertain, set 'unknown' or omit optional fields. "
    "Do not invent measurements; only estimate size_mm if the image clearly suggests scale.\n\n"
    "JSON Schema:\n" + json.dumps(EHR_JSON_SCHEMA["schema"], ensure_ascii=False)
)