# image_to_text_generator.py

import os, json, base64, time, csv
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch

# config file
from config_i2t_generator import GEN_CFG, SYSTEM_PROMPT

# ====================================================
# 1. path and settings
# ====================================================
ROOT = GEN_CFG["root_dir"]
IMG_DIR = os.path.join(ROOT, "images")
SPLIT_DIR = os.path.join(ROOT, "splits")

# Split Mode
if GEN_CFG["split_mode"] == "part0":
    OUT_SUFFIX = "_part0"
elif GEN_CFG["split_mode"] == "part1":
    OUT_SUFFIX = "_part1"
else:
    OUT_SUFFIX = ""  # full

print(f" Mode: {GEN_CFG['model_type']} | Split: {GEN_CFG['split_mode']} | Suffix: {OUT_SUFFIX}")

# ====================================================
# 2. model init
# ====================================================
model_engine = None
processor = None
client = None

if GEN_CFG["model_type"] == "openai":
    from openai import OpenAI
    # API Key setting (in config)
    api_key = GEN_CFG["openai"].get("api_key") 
    if not api_key or "YOUR_OPENAI" in api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    print(" OpenAI Client Initialized")

elif GEN_CFG["model_type"] == "qwen":
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    print(" Loading Qwen model... (this may take a while)")
    model_id = GEN_CFG["qwen"]["model_id"]
    model_engine = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map=GEN_CFG["qwen"]["device_map"]
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(" Qwen 2.5-VL Loaded")

# ====================================================
# 3. Helper Functions
# ====================================================
def img_to_data_url(path):
    """Base64 encoding for OpenAI """
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/jpeg;base64," + b64

def build_user_text(meta):
    return "\n".join([
        "Metadata:",
        f"- age: {meta.get('age')}",
        f"- sex: {meta.get('sex')}",
        f"- localization: {meta.get('localization')}",
        f"- dataset_label_dx (reference): {meta.get('dx')}",
    ])

def parse_json_output(content):
    """handle the result from LLM only with code block (```json ... ```)"""
    try:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        raise RuntimeError(f"JSON Parsing Failed: {e}")

# ----------------------------------------------------
# (A) OpenAI call function
# ----------------------------------------------------
def call_openai(img_path, meta):
    if not os.path.exists(img_path): return None, f"Image not found: {img_path}"
    
    cfg = GEN_CFG["openai"]
    data_url = img_to_data_url(img_path)
    
    for attempt in range(1, cfg["max_retries"]+1):
        try:
            resp = client.chat.completions.create(
                model=cfg["model_name"],
                messages=[
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content":[
                        {"type":"text","text": build_user_text(meta)},
                        {"type":"image_url","image_url":{"url": data_url}}
                    ]}
                ],
                response_format={"type":"json_object"},
                temperature=0
            )
            out = json.loads(resp.choices[0].message.content)
            return out, None
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            if attempt < cfg["max_retries"]:
                time.sleep(cfg["sleep_base"] * (2 ** (attempt - 1)))
            else:
                return None, err_msg

# ----------------------------------------------------
# (B) Qwen call function
# ----------------------------------------------------
def call_qwen(img_path, meta):
    if not os.path.exists(img_path): return None, f"Image not found: {img_path}"
    
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": build_user_text(meta)}
            ]}
        ]
        
        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to("cuda")
        
        # Generate
        with torch.inference_mode():
            generated_ids = model_engine.generate(
                **inputs, max_new_tokens=GEN_CFG["qwen"]["max_new_tokens"]
            )
        
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        
        # Parse JSON
        ehr = parse_json_output(output_text)
        return ehr, None

    except Exception as e:
        return None, f"Qwen Error: {e}"

# ====================================================
# 4. Main Execution Logic
# ====================================================
def get_split_data(df, mode):
    n = len(df)
    if mode == "part0":
        return df.iloc[: n // 2].reset_index(drop=True)
    elif mode == "part1":
        return df.iloc[n // 2 :].reset_index(drop=True)
    else:
        return df

def _get_processed_ids(ndjson_path):
    ids = set()
    if os.path.exists(ndjson_path):
        with open(ndjson_path, "r", encoding="utf-8") as rf:
            for line in rf:
                try:
                    rec = json.loads(line)
                    if "image_id" in rec: ids.add(str(rec["image_id"]))
                except: pass
    return ids

def _append_summary(csv_path, rows):
    if not rows: return
    exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists: w.writeheader()
        w.writerows(rows)

def run_pipeline():
    # 1) Load Data
    splits = ["train", "val", "test"]
    dfs = {}
    for s in splits:
        path = os.path.join(SPLIT_DIR, f"{s}.csv")
        full_df = pd.read_csv(path)
        dfs[s] = get_split_data(full_df, GEN_CFG["split_mode"])
    
    # Scanning available images
    print(f" Scanning images in {IMG_DIR}...")
    if not os.path.exists(IMG_DIR):
        print(f" Error: Image directory not found: {IMG_DIR}")
        return
    available_images = set(os.listdir(IMG_DIR)) # Scanning images
    print(f" Found {len(available_images)} files in image directory.")
    

    # 2) Process Each Split
    agg_ok, agg_err = 0, 0
    
    for split_name, df in dfs.items():
        # Output paths determined by model type + suffix
        folder = "ehrs" # folder name
        prefix = "qwen_" if GEN_CFG["model_type"] == "qwen" else "" # naming
        
        ndjson_path = os.path.join(ROOT, folder, f"{prefix}ehr_outputs_{split_name}{OUT_SUFFIX}.ndjson")
        summary_csv = os.path.join(ROOT, folder, f"{prefix}ehr_summary_{split_name}{OUT_SUFFIX}.csv")
        os.makedirs(os.path.dirname(ndjson_path), exist_ok=True)
        
        # Resume logic
        processed_ids = set() if GEN_CFG["force_reprocess"] else _get_processed_ids(ndjson_path)
        
        # todo_df = df[~df["image_id"].astype(str).isin(processed_ids)]
        # 
        # filtering with exisitng images only
        todo_mask = df["image_id"].astype(str).apply(
            lambda x: (x not in processed_ids) and (f"{x}.jpg" in available_images)
        )
        todo_df = df[todo_mask]
        
        print(f"\n[{split_name.upper()}{OUT_SUFFIX}]")
        print(f"  - Total: {len(df)} | Done: {len(processed_ids)}")
        print(f"  - To Process (Valid Images): {len(todo_df)}")
        # 
        
        print(f"\n[{split_name.upper()}{OUT_SUFFIX}] Total: {len(df)} | Done: {len(processed_ids)} | Todo: {len(todo_df)}")
        
        ok, err = 0, 0
        batch_rows = []
        
        with open(ndjson_path, "a", encoding="utf-8") as fout, \
             tqdm(total=len(todo_df), desc=f"{GEN_CFG['model_type']} processing") as pbar:
            
            for _, row in todo_df.iterrows():
                image_id = str(row["image_id"])
                img_path = os.path.join(IMG_DIR, image_id + ".jpg")
                
                meta = {
                    "age": int(row["age"]) if pd.notna(row["age"]) else None,
                    "sex": str(row["sex"]).lower() if pd.notna(row["sex"]) else "unknown",
                    "localization": str(row["localization"]).lower() if pd.notna(row["localization"]) else "unknown",
                    "dx": str(row["dx"]) if pd.notna(row["dx"]) else None
                }
                
                # Call specific model
                if GEN_CFG["model_type"] == "openai":
                    ehr, error_msg = call_openai(img_path, meta)
                else:
                    ehr, error_msg = call_qwen(img_path, meta)
                
                # Result Handling
                res_row = {
                    "image_id": image_id, 
                    "split": split_name, 
                    "dx_true": meta["dx"]
                }
                
                if ehr and not error_msg:
                    # success case
                    fout.write(json.dumps({"image_id": image_id, "ehr": ehr}, ensure_ascii=False) + "\n")
                    ok += 1
                    assess = ehr.get("assessment", {})
                    res_row.update({
                        "pred_label": assess.get("provisional_diagnosis_label"),
                        "malignancy_risk": assess.get("malignancy_risk"),
                        "error": ""
                    })
                else:
                    # fail case
                    fout.write(json.dumps({"image_id": image_id, "error": error_msg}, ensure_ascii=False) + "\n")
                    err += 1
                    res_row.update({
                        "pred_label": None, 
                        "malignancy_risk": None, 
                        "error": str(error_msg)
                    })
                
                batch_rows.append(res_row)
                
                # Checkpoint Save
                if len(batch_rows) >= GEN_CFG["checkpoint_every"]:
                    fout.flush()
                    _append_summary(summary_csv, batch_rows)
                    batch_rows = []
                
                pbar.update(1)
                pbar.set_postfix(ok=ok, err=err)
        
        # Final flush for the split
        _append_summary(summary_csv, batch_rows)
        print(f"Finished {split_name}: OK={ok}, ERR={err} -> Saved to {ndjson_path}")
        agg_ok += ok
        agg_err += err

    print(f"\n All Jobs Done. Total OK: {agg_ok}, Total ERR: {agg_err}")

if __name__ == "__main__":
    run_pipeline()