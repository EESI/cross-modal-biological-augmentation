CFG = {
    "mode": "ehr",   # "ehr" / "empo3" / "both"

    "ehr": {
        "base_dir": "/path/to/ehr_dataset/",
        "out_dir": "/content",
        "backbone_name": "convnextv2_large.fcmae_ft_in22k_in1k",
        # "gated" / "concat" / "image_only" / "meta_only"
        "fusion_type": "concat",
        "img_size": 384,
        "batch_size": 24,
        "epochs_stage1": 40,
        "epochs_stage2": 7,
        "seed": 0,

        # TTA option
        "use_tta": False,
        "tta_n": 1,

        # ensemble option
        "use_mlp_ensemble": True,
        "ensemble_weights": (0.6, 0.4),#(0.7, 0.3),  # (fusion, mlp)

        # optimizer selection: "adamw" or "lion"
        "optimizer": "adamw",

        # Mixup / mixup-like option 
        "use_mixup": True,
        "mixup_epochs": 24, #10,
        "mixup_alpha": 0.3,
        "cutmix_alpha": 1.0,
        "mixup_prob": 0.6,

        # EMA option
        "use_ema": True,
        "ema_decay": 0.9995, #0.9999,

        # Stage-2 debias on/off
        "use_stage2": True,

        # Calibration option 
        "use_bias_calibration": True,
        "tau_grid": (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
        "bias_iters": 3,

        
    },

    "empo3": {
        "base_dir": "/path/to/empo_dataset/",
        "out_dir": "/content",
        "backbone_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        # "gated" / "concat" / "image_only" / "tab_only"
        "fusion_type": "gated",
        "img_size": 224,
        "batch_size": 32,
        "epochs_stage1": 30,
        "epochs_stage2": 7,
        "seed": 42,

        # TTA option
        "use_tta": True,
        "tta_n": 8,

        # ensemble option
        "use_mlp_ensemble": True,
        "ensemble_weights": (0.5, 0.5),  # (fusion, mlp)

        # optimizer selection: "adamw" or "lion"
        "optimizer": "lion",

        # Mixup / mixup-like option 
        "use_mixup": True,
        "mixup_epochs": 30,    
        "mixup_alpha": 0.4,
        "cutmix_alpha": 0.0,  
        "mixup_prob": 0.3,

        # EMA option 
        "use_ema": False,
        "ema_decay": 0.9999,

        # Stage-2 on/off
        "use_stage2": True,

        # Calibration option 
        "use_bias_calibration": False,
        "tau_grid": (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
        "bias_iters": 0,
    }
}
