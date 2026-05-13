# cross-modal-biological-augmentation


Official supplementary code for:
"Cross-Modal Generative Augmentation for Multimodal Biological Classification"

This framework supports cross-modal generative augmentation pipeline training for both **EHR (Electronic Health Records) integration tasks** and **EMPO500 environmental classification tasks**.


## Project Structure


The code is organized as follows:

```text
.
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── code/                       # Source code directory
    ├── main.py                 # Entry point for training and evaluation
    ├── config.py               # Configuration for paths, hyperparameters, and modes
    ├── pipeline.py   # Core pipeline implementation    
    ├── utils.py                # Utility functions (seed setting, device configuration)
    │   # --- Data Generation Scripts ---
    ├── image_to_text_generator.py  # Script for generating EHRs from images (I2T)
    ├── config_i2t_generator.py     # Configuration for I2T (OpenAI / Qwen-VL)
    ├── text_to_image_generator.py  # Script for generating synthetic images from text (T2I)
    └── config_t2i_generator.py     # Configuration for T2I (Gemini / Qwen-Image)
```

## Environment Setup

The code requires **Python 3.8+** and **PyTorch**. You can install the necessary dependencies using the command below:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**

  * `torch`, `torchvision`
  * `timm`, `transformers`, `diffusers`, `accelerate`
  * `lion-pytorch` (Lion optimizer)
  * `torchmetrics`
  * `scikit-learn`, `pandas`, `numpy`
  * `google-genai` (for Gemini API), `openai` (for OpenAI API)

  






## Usage

### 1\. Configuration

All hyperparameters, file paths, and model settings are defined in `config.py`.
Before running the code, please ensure the `base_dir` in `config.py` points to the directory containing your dataset.

  * **EHR Task:** Configure paths under `CFG["ehr"]`
  * **EMPO500 Task:** Configure paths under `CFG["empo500"]`

### 2\. Running the Code

You can run the training and evaluation pipeline using `main.py`. The execution mode is controlled by the `CFG["mode"]` variable in `config.py`.

To run the code:

```bash
cd code
python main.py
```

### 3\. Switching Modes

To switch between tasks, modify the `mode` variable in `config.py`:

```python
# inside config.py
CFG = {
    "mode": "ehr",    # Options: "ehr", "empo500", "both"
    ...
}
```

## Data Preparation

Please organize your data directory as expected by the `config.py`.

**For EHR Task:**

  * Expects `images/` directory.
  * Expects `splits/` directory containing `train.csv`, `val.csv`, `test.csv`.
  * Expects `ehrs/` directory containing NDJSON files (e.g., `qwen_ehr_outputs_*.ndjson`).

**For EMPO500 Task:**

  * Expects `images_qwen/` directory.
  * Expects metadata CSV files (e.g., `meta_train_envfeature.csv`, `meta_val_envfeature.csv`, `meta_test_envfeature.csv`) in the base directory.



## Usage: Data Generation Pipelines

We provide standalone scripts to reproduce the synthetic data generation processes described in the paper. These scripts support both API-based (OpenAI/Gemini) and Local GPU-based (Qwen) models.

### 1\. Image-to-Text (EHR Generation)

This script generates structured EHR data (JSON) from dermoscopic images.

  * **Configuration:** Edit `code/config_i2t_generator.py`.
      * Select model: `"model_type": "openai"` or `"qwen"`.
      * Set split mode: `"part0"`, `"part1"`, or `"full"`.
  * **Run:**
    ```bash
    python code/image_to_text_generator.py
    ```

### 2\. Text-to-Image (Synthetic EMPO Generation)

This script generates photorealistic environmental images based on metadata prompts (Biome, Material, etc.).

  * **Configuration:** Edit `code/config_t2i_generator.py`.
      * Select model: `"model_type": "gemini"` or `"qwen"`.
      * Set output directories and API keys (if using Gemini).
  * **Run:**
    ```bash
    python code/text_to_image_generator.py
    ```




## Project Links

- GitHub: https://github.com/EESI/cross-modal-biological-augmentation
- HuggingFace: coming soon






