# ==================================================
# Unified script: EHR (HAM10000) + EMPO500 + MLP Ensemble
#   - CFG["mode"] = "ehr" / "empo500" / "both"
# ==================================================

import os, json, math, re, random
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import timm
from timm.utils import ModelEmaV2
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchmetrics.classification import (
    MulticlassAUROC, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
)
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils import device, set_seed

# ======================================================================
# 1) EHR + HAM10000 pipeline
# ======================================================================

def run_ehr(ehr_cfg):
    print("=" * 60)
    print("EHR + HAM10000 pipeline")
    print("=" * 60)
    set_seed(ehr_cfg.get("seed", 0))

    # =========================
    # 0) Paths / CSV load
    # =========================
    ROOT = ehr_cfg["base_dir"]
    IMG_DIR = os.path.join(ROOT, "images")
    SPLIT_DIR = os.path.join(ROOT, "splits")
    os.makedirs(SPLIT_DIR, exist_ok=True)

    train_csv = os.path.join(SPLIT_DIR, "train.csv")
    val_csv = os.path.join(SPLIT_DIR, "val.csv")
    test_csv = os.path.join(SPLIT_DIR, "test.csv")

    try:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
    except FileNotFoundError as e:
        print("[EHR] Error loading CSV files:", e)
        raise

    # -------------------------
    # Label encoder (7-class)
    # -------------------------
    le = LabelEncoder()
    le.fit(train_df["dx"])
    num_classes = len(le.classes_)
    print("[EHR] Classes:", list(le.classes_))

    # =========================
    # 1) EHR NDJSON loading
    # =========================
    def _ehr_paths(split_name):
        return [
            os.path.join(ROOT, f"ehrs/qwen_ehr_outputs_{split_name}_part0.ndjson"),
            os.path.join(ROOT, f"ehrs/qwen_ehr_outputs_{split_name}_part1.ndjson"),
        ]

    def load_ehr_map_for_split(split_name):
        """return: dict image_id -> ehr(dict) or None if error/missing"""
        ehr_map = {}
        for p in _ehr_paths(split_name):
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    iid = str(rec.get("image_id"))
                    if not iid:
                        continue
                    if "ehr" in rec and isinstance(rec["ehr"], dict):
                        # label error protection
                        ass = rec["ehr"].get("assessment")
                        if isinstance(ass, dict):
                            ass.pop("provisional_diagnosis_label", None)
                        ehr_map[iid] = rec["ehr"]
                    else:
                        ehr_map[iid] = None
        return ehr_map

    ehr_train = load_ehr_map_for_split("train")
    ehr_val = load_ehr_map_for_split("val")

    def ensure_all_ids(df, emap):
        for iid in df["image_id"].astype(str):
            if iid not in emap:
                emap[iid] = None

    ensure_all_ids(train_df, ehr_train)
    ensure_all_ids(val_df, ehr_val)

    print("[EHR] loaded:",
          f"train {sum(e is not None for e in ehr_train.values())}/{len(ehr_train)}",
          f"val {sum(e is not None for e in ehr_val.values())}/{len(ehr_val)}")

    # -------------------------
    # 1-1) vocab, scaler from train EHR
    # -------------------------
    _token_pat = re.compile(r"[a-z]+")
    def simple_tokens(s: str):
        s = (s or "").lower()
        return _token_pat.findall(s)

    def get_in(ehr, path, default=None):
        cur = ehr
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    sex_vocab = ["unknown"]
    enc_type_vocab = ["unknown"]
    site_vocab = ["unknown"]
    anat_vocab = ["unknown"]
    order_vocab = ["biopsy", "dermoscopy_followup", "reassurance", "urgent_referral", "other"]
    vf_counter = Counter()

    ages, sizes = [], []

    for iid, ehr in ehr_train.items():
        if ehr is None:
            continue
        # numeric
        age = get_in(ehr, "patient.age_years", None)
        if isinstance(age, (int, float)):
            ages.append(float(age))
        size_mm = get_in(ehr, "lesion_observation.size_mm", None)
        if isinstance(size_mm, (int, float)):
            sizes.append(float(size_mm))
        # categorical
        sex = str(get_in(ehr, "patient.sex", "unknown")).strip().lower() or "unknown"
        etype = str(get_in(ehr, "encounter.encounter_type", "unknown")).strip().lower() or "unknown"
        site = str(get_in(ehr, "encounter.site", "unknown")).strip().lower() or "unknown"
        anat = str(get_in(ehr, "lesion_observation.anatomical_site", "unknown")).strip().lower() or "unknown"

        if sex not in sex_vocab:
            sex_vocab.append(sex)
        if etype not in enc_type_vocab:
            enc_type_vocab.append(etype)
        if site not in site_vocab:
            site_vocab.append(site)
        if anat not in anat_vocab:
            anat_vocab.append(anat)

        # visual findings
        vfs = get_in(ehr, "lesion_observation.visual_findings", [])
        if isinstance(vfs, list):
            for s in vfs:
                for t in simple_tokens(str(s)):
                    vf_counter[t] += 1

    def _fit_z(xs):
        if len(xs) == 0:
            return 0.0, 1.0
        mu = float(np.mean(xs))
        sd = float(np.std(xs))
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        return mu, sd

    AGE_MU, AGE_SD = _fit_z(ages)
    SIZE_MU, SIZE_SD = _fit_z(sizes)

    VF_TOPK = 64
    vf_vocab = [w for w, _ in vf_counter.most_common(VF_TOPK)]

    sex2idx = {w: i for i, w in enumerate(sex_vocab)}
    enc2idx = {w: i for i, w in enumerate(enc_type_vocab)}
    site2idx = {w: i for i, w in enumerate(site_vocab)}
    anat2idx = {w: i for i, w in enumerate(anat_vocab)}
    order2idx = {w: i for i, w in enumerate(order_vocab)}
    vf2idx = {w: i for i, w in enumerate(vf_vocab)}

    SEX_DIM = len(sex_vocab)
    ENC_DIM = len(enc_type_vocab)
    SITE_DIM = len(site_vocab)
    ANAT_DIM = len(anat_vocab)
    ORD_DIM = len(order_vocab)
    VF_DIM = len(vf_vocab)

    print(f"[EHR vocabs] sex={SEX_DIM}, enc_type={ENC_DIM}, site={SITE_DIM}, anat={ANAT_DIM}, "
          f"orders={ORD_DIM}, vf_top={VF_DIM}")
    print(f"[EHR scalers] age_mu={AGE_MU:.2f} sd={AGE_SD:.2f} | size_mu={SIZE_MU:.2f} sd={SIZE_SD:.2f}")

    META_EHR_DIM = 3 + SEX_DIM + ENC_DIM + SITE_DIM + ANAT_DIM + ORD_DIM + VF_DIM
    print("[EHR] META_EHR_DIM =", META_EHR_DIM)

    def ehr_to_vec(ehr):
        # numeric
        age = get_in(ehr, "patient.age_years", None)
        age_z = (float(age) - AGE_MU) / AGE_SD if isinstance(age, (int, float)) else 0.0
        size = get_in(ehr, "lesion_observation.size_mm", None)
        size_z = (float(size) - SIZE_MU) / SIZE_SD if isinstance(size, (int, float)) else 0.0
        risk = get_in(ehr, "assessment.malignancy_risk", 0.0)
        try:
            risk = float(risk)
        except Exception:
            risk = 0.0

        # categorical → one/multi-hot
        sex = (str(get_in(ehr, "patient.sex", "unknown")).strip().lower() or "unknown")
        etype = (str(get_in(ehr, "encounter.encounter_type", "unknown")).strip().lower() or "unknown")
        site = (str(get_in(ehr, "encounter.site", "unknown")).strip().lower() or "unknown")
        anat = (str(get_in(ehr, "lesion_observation.anatomical_site", "unknown")).strip().lower() or "unknown")

        sex_oh = np.zeros(SEX_DIM, dtype=np.float32)
        sex_oh[sex2idx.get(sex, 0)] = 1.0
        enc_oh = np.zeros(ENC_DIM, dtype=np.float32)
        enc_oh[enc2idx.get(etype, 0)] = 1.0
        site_oh = np.zeros(SITE_DIM, dtype=np.float32)
        site_oh[site2idx.get(site, 0)] = 1.0
        anat_oh = np.zeros(ANAT_DIM, dtype=np.float32)
        anat_oh[anat2idx.get(anat, 0)] = 1.0

        ord_oh = np.zeros(ORD_DIM, dtype=np.float32)
        orders = get_in(ehr, "orders", [])
        if isinstance(orders, list):
            for od in orders:
                if not isinstance(od, dict):
                    continue
                t = str(od.get("type", "")).strip().lower()
                if t in order2idx:
                    ord_oh[order2idx[t]] = 1.0

        vf_oh = np.zeros(VF_DIM, dtype=np.float32)
        vfs = get_in(ehr, "lesion_observation.visual_findings", [])
        if isinstance(vfs, list):
            for s in vfs:
                for t in simple_tokens(str(s)):
                    j = vf2idx.get(t, None)
                    if j is not None:
                        vf_oh[j] = 1.0

        head = np.array([age_z, size_z, risk], dtype=np.float32)
        return np.concatenate([head, sex_oh, enc_oh, site_oh, anat_oh, ord_oh, vf_oh], axis=0)

    # =========================
    # 2) Dataset + loaders
    # =========================
    class HamDatasetMeta(Dataset):
        def __init__(self, df, img_dir, label_encoder, ehr_map, is_train=False, size=384):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.le = label_encoder
            self.size = size
            self.ehr_map = ehr_map

            if is_train:
                self.tf = T.Compose([
                    T.RandomResizedCrop(size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ColorJitter(0.15, 0.15, 0.15, 0.07),
                    T.RandomAffine(degrees=15, translate=(0.05, 0.05)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    T.RandomErasing(p=0.35, scale=(0.02, 0.1), ratio=(0.3, 3.3), value="random"),
                ])
            else:
                self.tf = T.Compose([
                    T.Resize(int(size * 1.05)),
                    T.CenterCrop(size),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            iid = str(row["image_id"])
            img = Image.open(os.path.join(self.img_dir, iid + ".jpg")).convert("RGB")
            x = self.tf(img)
            y = self.le.transform([row["dx"]])[0]

            ehr = self.ehr_map.get(iid, None)
            if ehr is None:
                meta_vec = np.zeros(META_EHR_DIM, dtype=np.float32)
            else:
                meta_vec = ehr_to_vec(ehr)

            return x, torch.tensor(y, dtype=torch.long), torch.from_numpy(meta_vec)

    def collate_fn_meta(batch):
        xs, ys, metas = zip(*batch)
        return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long), torch.stack(metas, 0).float()

    IMG_SIZE = ehr_cfg["img_size"]
    BATCH = ehr_cfg["batch_size"]
    NUM_WORKERS = 4

    train_targets = torch.tensor(le.transform(train_df["dx"]), dtype=torch.long)
    class_counts = np.bincount(train_targets.numpy(), minlength=num_classes)
    print("[EHR] class_counts:", dict(zip(le.classes_, class_counts)))

    weights_per_class = 1.0 / (class_counts + 1e-6)
    sample_weights = weights_per_class[train_targets.numpy()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        HamDatasetMeta(train_df, IMG_DIR, le, ehr_train, is_train=True, size=IMG_SIZE),
        batch_size=BATCH, sampler=sampler, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=(NUM_WORKERS > 0), prefetch_factor=2,
        collate_fn=collate_fn_meta, drop_last=True
    )
    val_loader = DataLoader(
        HamDatasetMeta(val_df, IMG_DIR, le, ehr_val, is_train=False, size=IMG_SIZE),
        batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=(NUM_WORKERS > 0), prefetch_factor=2,
        collate_fn=collate_fn_meta
    )
    ehr_test = load_ehr_map_for_split("test")
    ensure_all_ids(test_df, ehr_test)
    test_loader = DataLoader(
        HamDatasetMeta(test_df, IMG_DIR, le, ehr_test, is_train=False, size=IMG_SIZE),
        batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=(NUM_WORKERS > 0), prefetch_factor=2,
        collate_fn=collate_fn_meta
    )

    # ---- EHR meta matrix for MLP ensemble ----
    def collect_meta_matrix(df, ehr_map, label_encoder):
        X_meta, y_meta = [], []
        for _, row in df.iterrows():
            iid = str(row["image_id"])
            ehr = ehr_map.get(iid, None)
            if ehr is None:
                vec = np.zeros(META_EHR_DIM, dtype=np.float32)
            else:
                vec = ehr_to_vec(ehr)
            X_meta.append(vec)
            y_meta.append(row["dx"])
        X_meta = np.stack(X_meta, axis=0)
        y_meta = label_encoder.transform(y_meta)
        return X_meta, y_meta

    X_train_meta, y_train_meta = collect_meta_matrix(train_df, ehr_train, le)
    X_val_meta,   y_val_meta   = collect_meta_matrix(val_df,   ehr_val,   le)
    X_test_meta,  y_test_meta  = collect_meta_matrix(test_df,  ehr_test,  le)

    # =========================
    # 3) Model (with configurable fusion)
    # =========================
    BACKBONE_NAME = ehr_cfg["backbone_name"]
    FUSION_TYPE = ehr_cfg["fusion_type"]

    img_backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0).to(device)
    img_feat_dim = img_backbone.num_features
    print(f"[EHR] Using backbone: {BACKBONE_NAME} (feat dim={img_feat_dim}), fusion_type={FUSION_TYPE}")

    class MetaEHRMLP(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, drop: float = 0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden, out_dim),
                nn.GELU(),
            )
        def forward(self, m: torch.Tensor) -> torch.Tensor:
            return self.net(m)

    meta_enc = MetaEHRMLP(META_EHR_DIM, out_dim=img_feat_dim, hidden=512, drop=0.15).to(device)

    class FusionNet(nn.Module):
        def __init__(self, img_b, meta_e, num_classes, fusion_type="concat"):
            super().__init__()
            self.img_b = img_b
            self.meta_e = meta_e
            self.fusion_type = fusion_type

            img_dim = self.img_b.num_features

            if fusion_type == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(img_dim * 2, img_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(img_dim, img_dim),
                    nn.Sigmoid()
                )
                head_in_dim = img_dim
            elif fusion_type == "concat":
                self.gate = None
                head_in_dim = img_dim * 2
            elif fusion_type == "image_only":
                self.gate = None
                head_in_dim = img_dim
            elif fusion_type == "meta_only":
                self.gate = None
                head_in_dim = img_dim
            else:
                raise ValueError(f"Unknown EHR fusion_type: {fusion_type}")

            self.head = nn.Sequential(
                nn.Linear(head_in_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        def forward(self, x, meta, use_meta=True):
            img_z = self.img_b(x)
            meta_z = self.meta_e(meta) if use_meta else torch.zeros_like(img_z)

            if self.fusion_type == "gated":
                gate = self.gate(torch.cat([img_z, meta_z], dim=-1))
                fused = gate * img_z + (1.0 - gate) * meta_z
            elif self.fusion_type == "concat":
                fused = torch.cat([img_z, meta_z], dim=-1)
            elif self.fusion_type == "image_only":
                fused = img_z
            elif self.fusion_type == "meta_only":
                fused = meta_z
            else:
                raise ValueError(f"Unknown EHR fusion_type: {self.fusion_type}")

            return self.head(fused)

    fusion_net = FusionNet(img_backbone, meta_enc, num_classes, fusion_type=FUSION_TYPE).to(device)

    # EMA 옵션
    use_ema = ehr_cfg.get("use_ema", True)
    ema_decay = ehr_cfg.get("ema_decay", 0.9999)
    ema = ModelEmaV2(fusion_net, decay=ema_decay, device=device) if use_ema else None

    # =========================
    # 4) Loss / Optimizer / Scheduler
    # =========================
    def get_cb_weights(counts, beta=0.999):
        counts = np.asarray(counts, dtype=np.float64)
        eff = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / np.maximum(eff, 1e-8)
        w = w / w.sum() * len(counts)
        return torch.tensor(w, dtype=torch.float32, device=device)

    cb_weights = get_cb_weights(class_counts, beta=0.999)

    class CBFocalLoss(nn.Module):
        def __init__(self, gamma=1.5):
            super().__init__()
            self.gamma = gamma
        def forward(self, logits, target, class_weights=None):
            log_prob = F.log_softmax(logits, dim=1)
            ce = F.nll_loss(log_prob, target, reduction="none")
            pt = torch.exp(-ce)
            focal = (1 - pt) ** self.gamma * ce
            if class_weights is not None:
                focal = focal * class_weights[target]
            return focal.mean()

    class BalancedSoftmaxCELoss(nn.Module):
        def __init__(self, samples_per_class):
            super().__init__()
            spc = torch.tensor(samples_per_class, dtype=torch.float32)
            self.register_buffer("log_spc", torch.log(spc + 1e-12))
        def forward(self, logits, target):
            log_den = torch.logsumexp(logits + self.log_spc.unsqueeze(0), dim=1)
            z_y = logits.gather(1, target.view(-1, 1)).squeeze(1)
            loss = (-z_y + log_den).mean()
            return loss

    # Mixup settings (based on CFG )
    MIXUP = ehr_cfg.get("use_mixup", True)
    MIXUP_EPOCHS = ehr_cfg.get("mixup_epochs", 10)
    mixup_fn = Mixup(
        mixup_alpha=ehr_cfg.get("mixup_alpha", 0.3),
        cutmix_alpha=ehr_cfg.get("cutmix_alpha", 1.0),
        prob=ehr_cfg.get("mixup_prob", 0.6),
        switch_prob=0.0,
        mode="batch",
        label_smoothing=0.0,
        num_classes=num_classes
    )

    cb_focal = CBFocalLoss(gamma=1.5)
    bal_ce = BalancedSoftmaxCELoss(class_counts).to(device)

    backbone_lr, head_lr = 3e-5, 3e-4
    opt_name_ehr = ehr_cfg.get("optimizer", "adamw").lower()
    print(f"[EHR] Optimizer (stage1) = {opt_name_ehr}")

    if opt_name_ehr == "lion":
        opt = Lion([
            {"params": fusion_net.img_b.parameters(), "lr": backbone_lr},
            {"params": fusion_net.meta_e.parameters(), "lr": head_lr},
            {"params": fusion_net.head.parameters(), "lr": head_lr},
        ], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([
            {"params": fusion_net.img_b.parameters(), "lr": backbone_lr},
            {"params": fusion_net.meta_e.parameters(), "lr": head_lr},
            {"params": fusion_net.head.parameters(), "lr": head_lr},
        ], weight_decay=1e-4)

    EPOCHS_STAGE1 = ehr_cfg["epochs_stage1"]
    warmup_epochs = 4

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, (EPOCHS_STAGE1 - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # =========================
    # 5) Metrics / calibration
    # =========================
    def compute_metrics(logits_cpu, targets_cpu):
        probs = torch.softmax(logits_cpu, dim=-1)
        auc_v = MulticlassAUROC(num_classes=num_classes, average="macro")(probs, targets_cpu).item()
        preds = logits_cpu.argmax(1)
        f1_macro_v = MulticlassF1Score(num_classes=num_classes, average="macro")(preds, targets_cpu).item()
        f1_weighted_v = MulticlassF1Score(num_classes=num_classes, average="weighted")(preds, targets_cpu).item()
        acc_v = (preds == targets_cpu).float().mean().item()
        return auc_v, f1_macro_v, f1_weighted_v, acc_v

    def per_class_report(logits_cpu, targets_cpu):
        preds = logits_cpu.argmax(1)
        f1_each = MulticlassF1Score(num_classes=num_classes, average=None)(preds, targets_cpu).tolist()
        rec_each = MulticlassRecall(num_classes=num_classes, average=None)(preds, targets_cpu).tolist()
        pre_each = MulticlassPrecision(num_classes=num_classes, average=None)(preds, targets_cpu).tolist()
        return f1_each, rec_each, pre_each

    prior = torch.tensor(class_counts / class_counts.sum(), dtype=torch.float32, device=device)

    def apply_logit_adjustment(logits, tau=0.0, bias=None):
        z = logits
        if tau != 0.0:
            z = z - tau * prior.log()
        if bias is not None:
            z = z + bias.view(1, -1).to(z.device)
        return z

    def optimize_biases(val_logits_cpu, val_targets_cpu, iters=2, cand=(-1.0, -0.5, 0.0, 0.5, 1.0)):
        bias = np.zeros(num_classes, dtype=np.float32)
        best_f1 = compute_metrics(torch.tensor(val_logits_cpu), val_targets_cpu)[1]
        for _ in range(iters):
            improved = False
            for c in range(num_classes):
                best_b = bias[c]
                local_best = best_f1
                for delta in cand:
                    trial = bias.copy()
                    trial[c] = delta
                    z = torch.tensor(val_logits_cpu) + torch.tensor(trial).view(1, -1)
                    f1m = compute_metrics(z, val_targets_cpu)[1]
                    if f1m > local_best:
                        local_best = f1m
                        best_b = delta
                if best_b != bias[c]:
                    bias[c] = best_b
                    best_f1 = local_best
                    improved = True
            if not improved:
                break
        return torch.tensor(bias, dtype=torch.float32)

    # calibration settings
    use_bias_calibration = ehr_cfg.get("use_bias_calibration", True)
    tau_grid = ehr_cfg.get("tau_grid", (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0))
    bias_iters = ehr_cfg.get("bias_iters", 3)

    # =========================
    # 6) Train / eval loops
    # =========================
    def run_epoch(dl, train: bool, epoch: int, use_ema_eval=False, tau_for_eval=0.0, bias_vec=None):
        fusion_net.train(train)
        total_loss, n = 0.0, 0
        all_logits, all_targets = [], []
        use_mix = (train and MIXUP and (epoch <= MIXUP_EPOCHS))

        for bi, (x, y, meta) in enumerate(dl, 1):
            x, y, meta = x.to(device), y.to(device), meta.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            if use_mix and (x.size(0) % 2 == 1):
                x, y, meta = x[:-1], y[:-1], meta[:-1]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if use_mix:
                    x_mix, y_mix = mixup_fn(x, y)
                    logits = fusion_net(x_mix, meta, use_meta=True)
                    loss = SoftTargetCrossEntropy()(logits, y_mix)
                else:
                    logits = fusion_net(x, meta, use_meta=True)
                    loss = cb_focal(logits, y, class_weights=cb_weights)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(fusion_net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update(fusion_net)

            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs

            if not train:
                m = ema.module if (use_ema_eval and ema is not None) else fusion_net
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = m(x, meta, use_meta=True)
                    logits = apply_logit_adjustment(logits, tau_for_eval, bias_vec)

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

            if bi % 50 == 0:
                print(f"  .. batch {bi} / ~{len(dl)} done")

        avg_loss = total_loss / max(n, 1)
        logits = torch.cat(all_logits, 0)
        targets = torch.cat(all_targets, 0)
        auc_v, f1m_v, f1w_v, acc_v = compute_metrics(logits, targets)
        return avg_loss, acc_v, auc_v, f1m_v, f1w_v, logits, targets

    @torch.no_grad()
    def eval_loader(dl, tau=0.0, bias_vec=None, use_ema_eval=True):
        m = ema.module if (use_ema_eval and ema is not None) else fusion_net
        m.eval()
        all_logits, all_targets = [], []
        for x, y, meta in dl:
            x, y, meta = x.to(device), y.to(device), meta.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = m(x, meta, use_meta=True)
                logits = apply_logit_adjustment(logits, tau, bias_vec)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
        logits = torch.cat(all_logits, 0)
        targets = torch.cat(all_targets, 0)
        return compute_metrics(logits, targets), logits, targets

    @torch.no_grad()
    def sweep_tau_for_macro_f1(loader, taus=(0.0, 0.5, 1.0, 1.5, 2.0), bias_vec=None, use_ema_eval=True):
        best_tau, best_f1 = None, -1.0
        for t in taus:
            (auc, f1m, f1w, acc), _, _ = eval_loader(loader, tau=t, bias_vec=bias_vec, use_ema_eval=use_ema_eval)
            if f1m > best_f1:
                best_f1, best_tau = f1m, t
        return 0.0 if best_tau is None else best_tau

    # =========================
    # 7) Stage-1 train
    # =========================
    best_val_f1, patience, pcnt = -1, 8, 0
    # ckpt_path = os.path.join("/content", "qwen_fusion_img_meta_stage1_best.pt")
    out_dir = ehr_cfg.get("out_dir", os.path.join(ROOT, "checkpoints_ehr"))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "qwen_fusion_img_meta_stage1_best.pt")

    for ep in range(1, EPOCHS_STAGE1 + 1):
        tr = run_epoch(train_loader, train=True, epoch=ep, use_ema_eval=False)
        va = run_epoch(val_loader, train=False, epoch=ep, use_ema_eval=True, tau_for_eval=0.0)
        sch.step()

        print(f"[S1 {ep:02d}] train loss {tr[0]:.4f} acc {tr[1]:.3f} auc {tr[2]:.3f} "
              f"f1_macro {tr[3]:.3f} f1_weighted {tr[4]:.3f} | "
              f"val loss {va[0]:.4f} acc {va[1]:.3f} auc {va[2]:.3f} "
              f"f1_macro {va[3]:.3f} f1_weighted {va[4]:.3f}")

        score = va[3]
        if score > best_val_f1:
            best_val_f1, pcnt = score, 0
            torch.save({
                "fusion_net": fusion_net.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "classes": list(le.classes_),
                "ehr_meta": {
                    "sex_vocab": sex_vocab,
                    "enc_type_vocab": enc_type_vocab,
                    "site_vocab": site_vocab,
                    "anat_vocab": anat_vocab,
                    "order_vocab": order_vocab,
                    "vf_vocab": vf_vocab,
                    "age_mu": AGE_MU, "age_sd": AGE_SD,
                    "size_mu": SIZE_MU, "size_sd": SIZE_SD,
                },
                "backbone": BACKBONE_NAME,
                "fusion_type": FUSION_TYPE,
            }, ckpt_path)
        else:
            pcnt += 1
            if pcnt >= patience:
                print("[EHR] Early stop (stage-1).")
                break

    # Stage-1 best ckpt load
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        fusion_net.load_state_dict(ckpt["fusion_net"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
    except Exception as e:
        print("[EHR] Warning: Could not load Stage-1 checkpoint. Error:", e)
        ckpt = None

    # =========================
    # 8) Stage-2 debias fine-tune (option)
    # =========================
    use_stage2 = ehr_cfg.get("use_stage2", True)
    EPOCHS_STAGE2 = ehr_cfg["epochs_stage2"]
    # ckpt_path_s2 = os.path.join("/content", "qwen_fusion_img_meta_stage2_best.pt")
    ckpt_path_s2 = os.path.join(out_dir, "qwen_fusion_img_meta_stage2_best.pt")

    if use_stage2 and ckpt is not None:
        for p in fusion_net.img_b.parameters():
            p.requires_grad = False

        print(f"[EHR] Optimizer (stage2) = {opt_name_ehr}")
        if opt_name_ehr == "lion":
            opt2 = Lion([
                {"params": fusion_net.meta_e.parameters(), "lr": 2e-4},
                {"params": fusion_net.head.parameters(), "lr": 2e-4},
            ], weight_decay=5e-5)
        else:
            opt2 = torch.optim.AdamW([
                {"params": fusion_net.meta_e.parameters(), "lr": 2e-4},
                {"params": fusion_net.head.parameters(), "lr": 2e-4},
            ], weight_decay=5e-5)

        best_val_f1_s2 = -1
        for ep in range(1, EPOCHS_STAGE2 + 1):
            fusion_net.train(True)
            total_loss, n = 0.0, 0
            for bi, (x, y, meta) in enumerate(train_loader, 1):
                x, y, meta = x.to(device), y.to(device), meta.to(device)
                opt2.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = fusion_net(x, meta, use_meta=True)
                    loss = bal_ce(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(
                    list(fusion_net.meta_e.parameters()) + list(fusion_net.head.parameters()),
                    1.0
                )
                scaler.step(opt2)
                scaler.update()
                if ema is not None:
                    ema.update(fusion_net)
                total_loss += loss.item() * x.size(0)
                n += x.size(0)

            (auc_v, f1m_v, f1w_v, acc_v), _, _ = eval_loader(val_loader, tau=0.0, bias_vec=None, use_ema_eval=True)
            print(f"[S2 {ep:02d}] train loss {total_loss / max(1, n):.4f} | val acc {acc_v:.3f} auc {auc_v:.3f} "
                  f"macroF1 {f1m_v:.3f} wF1 {f1w_v:.3f}")
            if f1m_v > best_val_f1_s2:
                best_val_f1_s2 = f1m_v
                torch.save({
                    "fusion_net": fusion_net.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "meta": ckpt
                }, ckpt_path_s2)

        if os.path.exists(ckpt_path_s2):
            ckpt_final = torch.load(ckpt_path_s2, map_location=device)
            print("[EHR] Loaded best Stage-2 checkpoint from", ckpt_path_s2)
        else:
            ckpt_final = ckpt
            print("[EHR] Stage-2 checkpoint not found. Using Stage-1 best.")
    else:
        ckpt_final = ckpt
        if not use_stage2:
            print("[EHR] Skipping Stage-2; using best Stage-1 checkpoint.")
        else:
            print("[EHR] Stage-1 checkpoint missing; cannot run Stage-2.")

    if ckpt_final is not None:
        fusion_net.load_state_dict(ckpt_final["fusion_net"])
        if ema is not None and ckpt_final.get("ema") is not None:
            ema.load_state_dict(ckpt_final["ema"])

    # =========================
    # 9) Tau / bias tuning & final test (option)
    # =========================
    if use_bias_calibration:
        print("\n[EHR] Sweeping tau on validation to maximize macro F1 (EMA) ...")
        best_tau = sweep_tau_for_macro_f1(
            val_loader,
            taus=tau_grid,
            bias_vec=None,
            use_ema_eval=True
        )

        (_, _, _, _), val_logits, val_targets = eval_loader(
            val_loader, tau=best_tau, bias_vec=None, use_ema_eval=True
        )
        bias_vec = optimize_biases(
            val_logits, val_targets, iters=bias_iters,
            cand=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
        )
    else:
        print("\n[EHR] Skipping calibration (tau/bias); using tau=0, no bias.")
        best_tau = 0.0
        bias_vec = None

    (val_auc, val_f1m, val_f1w, val_acc), val_logits2, val_targets2 = eval_loader(
        val_loader, tau=best_tau, bias_vec=bias_vec, use_ema_eval=True
    )
    val_f1_each, val_rec_each, val_pre_each = per_class_report(val_logits2, val_targets2)
    print(f"\n[EHR] VAL (tau={best_tau:.2f} + bias) | acc {val_acc:.3f} auc {val_auc:.3f} "
          f"macroF1 {val_f1m:.3f} wF1 {val_f1w:.3f}")
    for c, f1, rc, pr in zip(le.classes_, val_f1_each, val_rec_each, val_pre_each):
        print(f"  {c:>5s} | F1 {f1:.3f}  R {rc:.3f}  P {pr:.3f}")

    (test_auc, test_f1m, test_f1w, test_acc), test_logits, test_targets = eval_loader(
        test_loader, tau=best_tau, bias_vec=bias_vec, use_ema_eval=True
    )
    test_f1_each, test_rec_each, test_pre_each = per_class_report(test_logits, test_targets)
    print(f"\n[EHR] TEST (tau={best_tau:.2f} + bias) | acc {test_acc:.3f} auc {test_auc:.3f} "
          f"macroF1 {test_f1m:.3f} wF1 {test_f1w:.3f}")
    for c, f1, rc, pr in zip(le.classes_, test_f1_each, test_rec_each, test_pre_each):
        print(f"  {c:>5s} | F1 {f1:.3f}  R {rc:.3f}  P {pr:.3f}")

    # =========================
    # 10) EHR MLP Ensemble (meta-only)
    # =========================
    if ehr_cfg.get("use_mlp_ensemble", False):
        print("\n[EHR] === Training meta-only MLP for ensemble ===")

        @torch.no_grad()
        def predict_proba_ehr_fusion(m, loader, tau, bias):
            m.eval()
            all_probs, all_y = [], []
            for x, y, meta in tqdm(loader, desc="FusionNet probs (EHR)", leave=False):
                x, meta = x.to(device), meta.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = m(x, meta, use_meta=True)
                    logits = apply_logit_adjustment(logits, tau, bias)
                    probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_y.append(y.numpy())
            return np.concatenate(all_probs, 0), np.concatenate(all_y, 0)

        fusion_model_for_ens = ema.module if (ema is not None) else fusion_net
        fusion_probs, y_true = predict_proba_ehr_fusion(
            fusion_model_for_ens, test_loader, best_tau, bias_vec
        )
        num_classes_ehr = fusion_probs.shape[1]

        X_meta_train_full = np.concatenate([X_train_meta, X_val_meta], axis=0)
        y_meta_train_full = np.concatenate([y_train_meta, y_val_meta], axis=0)

        mlp_ehr = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42
        )
        mlp_ehr.fit(X_meta_train_full, y_meta_train_full)
        mlp_probs = mlp_ehr.predict_proba(X_test_meta)

        if mlp_probs.shape[1] != num_classes_ehr:
            raise ValueError(f"[EHR] Ensemble: MLP probs dim {mlp_probs.shape[1]} != num_classes {num_classes_ehr}")

        w_fusion, w_mlp = ehr_cfg.get("ensemble_weights", (0.7, 0.3))
        ensemble_probs = w_fusion * fusion_probs + w_mlp * mlp_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        acc = accuracy_score(y_true, ensemble_preds)
        f1m = f1_score(y_true, ensemble_preds, average="macro")
        prec_m = precision_score(y_true, ensemble_preds, average="macro")
        rec_m  = recall_score(y_true, ensemble_preds, average="macro")

        classes = np.arange(num_classes_ehr)
        y_true_bin = label_binarize(y_true, classes=classes)
        auroc = roc_auc_score(
            y_true_bin, ensemble_probs,
            average="macro", multi_class="ovr"
        )

        print(f"\n[EHR-ENSEMBLE]  Accuracy:        {acc:.4f}")
        print(f"[EHR-ENSEMBLE]  Macro-F1:       {f1m:.4f}")
        print(f"[EHR-ENSEMBLE]  Macro-Precision:{prec_m:.4f}")
        print(f"[EHR-ENSEMBLE]  Macro-Recall:   {rec_m:.4f}")
        print(f"[EHR-ENSEMBLE]  Macro-AUROC:    {auroc:.4f}")

        print("\n=== EHR Ensemble Classification Report ===")
        print(classification_report(y_true, ensemble_preds, target_names=list(le.classes_)))


# ======================================================================
# 2) EMPO500 pipeline (fusion + optional MLP ensemble)
# ======================================================================

def run_empo500(empo_cfg):
    print("=" * 60)
    print("EMPO500 pipeline")
    print("=" * 60)
    set_seed(empo_cfg.get("seed", 42))

    # EMPO500 calibration error message
    if empo_cfg.get("use_bias_calibration", False):
        print("[EMPO500] use_bias_calibration=True but, "
              "tau/bias calibration is only for EHR pipeline. (not applicable for EMPO500 now)")

    base_dir = empo_cfg["base_dir"]
    img_root = os.path.join(base_dir, "images_qwen")

    splits = {
        "train": pd.read_csv(base_dir + "meta_train_envfeature.csv"),
        "val":   pd.read_csv(base_dir + "meta_val_envfeature.csv"),
        "test":  pd.read_csv(base_dir + "meta_test_envfeature.csv"),
    }

    # raw string version (for MLP) 
    raw_splits = {sp: df.copy() for sp, df in splits.items()}

    features = ['env_biome', 'env_material', 'sample_type', 'scientific_name', 'empo_3']
    target = 'env_feature'

    # -------------------------
    # Label encoding for FusionModel
    # -------------------------
    encoders = {}
    for col in features + [target]:
        le = LabelEncoder()
        all_vals = pd.concat(
            [splits["train"][col], splits["val"][col], splits["test"][col]], axis=0
        )
        le.fit(all_vals.astype(str))
        encoders[col] = le
        for sp in splits:
            splits[sp][col] = le.transform(splits[sp][col].astype(str))
    num_classes = len(encoders[target].classes_)
    print("[EMPO500] num_classes:", num_classes)

    # =========================
    # Dataset
    # =========================
    class EMPO500Dataset(Dataset):
        def __init__(self, df, img_dir, split, size=224, is_train=False):
            self.df = df.reset_index(drop=True)
            self.img_dir = os.path.join(img_dir, split)
            self.split = split
            self.is_train = is_train
            self.tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.8, 1.0)) if is_train else T.Resize((size, size)),
                T.RandomHorizontalFlip() if is_train else T.Lambda(lambda x: x),
                T.ColorJitter(0.25, 0.25, 0.25, 0.1) if is_train else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, f"{self.split}_{idx}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")
            img = Image.open(img_path).convert("RGB")
            x = self.tf(img)
            tab = torch.tensor(row[features].astype(int).values, dtype=torch.long)
            y = torch.tensor(int(row[target]), dtype=torch.long)
            return x, tab, y

    IMG_SIZE = empo_cfg["img_size"]
    BATCH = empo_cfg["batch_size"]

    datasets = {
        sp: EMPO500Dataset(splits[sp], img_root, sp, size=IMG_SIZE, is_train=(sp == "train"))
        for sp in splits
    }
    loaders = {
        sp: DataLoader(datasets[sp], batch_size=BATCH, shuffle=(sp == "train"),
                       num_workers=4, pin_memory=True)
        for sp in splits
    }

    # =========================
    # Model (with configurable fusion)
    # =========================
    class TabEncoder(nn.Module):
        def __init__(self, cat_sizes, emb_dim=128, out_dim=512, dropout=0.3):
            super().__init__()
            self.embeds = nn.ModuleList([nn.Embedding(sz, emb_dim) for sz in cat_sizes])
            self.mlp = nn.Sequential(
                nn.Linear(len(cat_sizes) * emb_dim, 512),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, out_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        def forward(self, x):
            emb = [emb_layer(x[:, i]) for i, emb_layer in enumerate(self.embeds)]
            z = torch.cat(emb, dim=1)
            z = z + 0.1 * torch.randn_like(z)  # robustness
            return self.mlp(z)

    class FusionModel(nn.Module):
        def __init__(self, num_classes, cat_sizes, backbone_name, fusion_type="gated"):
            super().__init__()
            self.fusion_type = fusion_type
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True, num_classes=0
            )
            self.tab_encoder = TabEncoder(cat_sizes)
            img_dim = self.backbone.num_features
            tab_dim = 512

            self.proj_tab = nn.Linear(tab_dim, img_dim)

            if fusion_type == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(img_dim * 2, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(1024, img_dim),
                    nn.Sigmoid()
                )
                head_in_dim = img_dim
            elif fusion_type == "concat":
                self.gate = None
                head_in_dim = img_dim * 2
            elif fusion_type == "image_only":
                self.gate = None
                head_in_dim = img_dim
            elif fusion_type == "tab_only":
                self.gate = None
                head_in_dim = img_dim
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")

            self.head = nn.Sequential(
                nn.Linear(head_in_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )

        def forward(self, img, tab):
            img_z = self.backbone(img)
            tab_z = self.tab_encoder(tab)
            tab_z = self.proj_tab(tab_z)

            if self.fusion_type == "gated":
                gate = self.gate(torch.cat([img_z, tab_z], dim=1))
                fused = gate * img_z + (1 - gate) * tab_z
            elif self.fusion_type == "concat":
                fused = torch.cat([img_z, tab_z], dim=1)
            elif self.fusion_type == "image_only":
                fused = img_z
            elif self.fusion_type == "tab_only":
                fused = tab_z
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

            return self.head(fused)

    cat_sizes = [len(encoders[f].classes_) for f in features]
    model = FusionModel(
        num_classes,
        cat_sizes,
        backbone_name=empo_cfg["backbone_name"],
        fusion_type=empo_cfg["fusion_type"]
    ).to(device)
    print("[EMPO500] Backbone:", empo_cfg["backbone_name"], "| fusion_type:", empo_cfg["fusion_type"])
    print("[EMPO500] Model params:",
          round(sum(p.numel() for p in model.parameters()) / 1e6, 2), "M")

    # EMA option (EMPO500)
    use_ema = empo_cfg.get("use_ema", False)
    ema_decay = empo_cfg.get("ema_decay", 0.9999)
    ema = ModelEmaV2(model, decay=ema_decay, device=device) if use_ema else None

    # =========================
    # Loss, Optimizer, Scheduler
    # =========================
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    backbone_params = list(model.backbone.parameters())
    fusion_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    opt_name_empo = empo_cfg.get("optimizer", "adamw").lower()
    print(f"[EMPO500] Optimizer = {opt_name_empo}")

    if opt_name_empo == "lion":
        opt = Lion([
            {"params": backbone_params, "lr": 2e-5},
            {"params": fusion_params, "lr": 1e-4}
        ], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": 2e-5},
            {"params": fusion_params, "lr": 1e-4}
        ], weight_decay=1e-4)

    scheduler = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)

    use_mixup = empo_cfg.get("use_mixup", True)
    mixup_prob = empo_cfg.get("mixup_prob", 0.3)
    mixup_alpha = empo_cfg.get("mixup_alpha", 0.4)

    # =========================
    # Train / Eval
    # =========================
    def train_one_epoch(model, loader):
        model.train()
        total = 0
        for img, tab, y in tqdm(loader, desc="Train", leave=False):
            img, tab, y = img.to(device), tab.to(device), y.to(device)
            if use_mixup and (random.random() < mixup_prob):  # mixup-like
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(img.size(0), device=img.device)
                img = lam * img + (1 - lam) * img[perm]
            opt.zero_grad()
            out = model(img, tab)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(model)
            total += loss.item() * img.size(0)
        return total / len(loader.dataset)

    @torch.no_grad()
    def evaluate(model, loader, tta=False, n_tta=8, use_ema_eval=True):
        base_model = ema.module if (use_ema_eval and ema is not None) else model
        base_model.eval()
        preds, trues = [], []
        for img, tab, y in tqdm(loader, desc="Eval", leave=False):
            img, tab = img.to(device), tab.to(device)
            if not tta:
                out = base_model(img, tab)
            else:
                out_sum = 0
                for _ in range(n_tta):
                    img_aug = T.ColorJitter(0.1, 0.1, 0.1)(img.cpu())
                    out_sum += base_model(img_aug.to(device), tab)
                out = out_sum / n_tta
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
        acc = accuracy_score(trues, preds)
        f1m = f1_score(trues, preds, average="macro")
        return acc, f1m

    # =========================
    # Stage-wise training
    # =========================
    out_dir = empo_cfg.get("out_dir", os.path.join(base_dir, "checkpoints_empo500"))
    os.makedirs(out_dir, exist_ok=True)

    # ckpt = os.path.join("/content", "convnextv2_gatedfusion_best.pt")
    ckpt_s1  = os.path.join(out_dir, "convnextv2_gatedfusion_stage1_best.pt")
    ckpt_s2  = os.path.join(out_dir, "convnextv2_gatedfusion_stage2_best.pt")
    # best_val = -1
    best_val_s1 = -1
    best_val_s2 = -1

    EPOCHS_STAGE1 = empo_cfg["epochs_stage1"]
    EPOCHS_STAGE2 = empo_cfg["epochs_stage2"]
    use_stage2 = empo_cfg.get("use_stage2", True)

    # --- Stage1 ---
    for p in model.backbone.parameters():
        p.requires_grad = False
    print(f"\n[EMPO500] ===== Stage1: train tab fusion only ({EPOCHS_STAGE1} epochs) =====")
    for ep in range(1, EPOCHS_STAGE1 + 1):
        loss = train_one_epoch(model, loaders["train"])
        val_acc, val_f1 = evaluate(model, loaders["val"], use_ema_eval=True)
        scheduler.step()
        print(f"[Stage1|Ep{ep:02d}] loss={loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_val_s1:
            # best_val = val_f1
            # torch.save(model.state_dict(), ckpt)
            # print(f"  Saved best (f1={val_f1:.4f})")
            best_val_s1 = val_f1
            torch.save(model.state_dict(), ckpt_s1)
            print(f"  [Stage1] Saved best (f1={val_f1:.4f}) -> {ckpt_s1}")

    # --- Stage2 ---
    if use_stage2:
        if os.path.exists(ckpt_s1):
            model.load_state_dict(torch.load(ckpt_s1, map_location=device))
            print(f"[EMPO500] Loaded Stage1 best checkpoint from {ckpt_s1} for Stage2 fine-tuning.")
        else:
            print("[EMPO500] Warning: Stage1 checkpoint not found, starting Stage2 from current model.")

        for p in model.backbone.parameters():
            p.requires_grad = True

        print(f"\n[EMPO500] ===== Stage2: fine-tune full model ({EPOCHS_STAGE2} epochs) =====")
        for ep in range(1, EPOCHS_STAGE2 + 1):
            loss = train_one_epoch(model, loaders["train"])
            val_acc, val_f1 = evaluate(model, loaders["val"], use_ema_eval=True)
            scheduler.step()
            print(f"[Stage2|Ep{ep:02d}] loss={loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            if val_f1 > best_val_s2:
                # best_val = val_f1
                # torch.save(model.state_dict(), ckpt)
                # print(f"  Saved best (f1={val_f1:.4f})")
                best_val_s2 = val_f1
                torch.save(model.state_dict(), ckpt_s2)
                print(f"  [Stage2] Saved best (f1={val_f1:.4f}) -> {ckpt_s2}")
    else:
        print("[EMPO500] Skipping Stage2 (use_stage2=False).")

    # =========================
    # Test (TTA Option)
    # =========================
    model.load_state_dict(torch.load(ckpt_s2, map_location=device))
    use_tta = empo_cfg["use_tta"]
    n_tta = empo_cfg["tta_n"]
    test_acc, test_f1 = evaluate(model, loaders["test"], tta=use_tta, n_tta=n_tta, use_ema_eval=True)
    print(f"\n[EMPO500]  Test Accuracy: {test_acc:.4f},  Test Macro-F1: {test_f1:.4f} "
          f"(TTA={use_tta}, n_tta={n_tta})")

    # =========================
    # Ensemble: FusionModel + MLP (tab-only)
    # =========================
    if empo_cfg.get("use_mlp_ensemble", False):
        print("\n[EMPO500] === Training tab-only MLP for ensemble ===")

        @torch.no_grad()
        def predict_proba_fusion(m, loader):
            base_model = ema.module if ema is not None else m
            base_model.eval()
            all_probs, all_y = [], []
            for img, tab, y in tqdm(loader, desc="FusionModel probs", leave=False):
                img, tab = img.to(device), tab.to(device)
                out = torch.softmax(base_model(img, tab), dim=1)
                all_probs.append(out.cpu().numpy())
                all_y.append(y.numpy())
            return np.concatenate(all_probs, axis=0), np.concatenate(all_y, axis=0)

        fusion_probs, y_true = predict_proba_fusion(model, loaders["test"])
        num_classes_local = fusion_probs.shape[1]

        le_target = encoders[target]
        X_train_tab = raw_splits["train"][features]
        y_train_tab = le_target.transform(raw_splits["train"][target].astype(str))

        X_test_tab  = raw_splits["test"][features]
        y_test_tab  = le_target.transform(raw_splits["test"][target].astype(str))

        categorical_features = features
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)]
        )

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', mlp)
        ])

        pipeline.fit(X_train_tab, y_train_tab)
        mlp_probs = pipeline.predict_proba(X_test_tab)

        if mlp_probs.shape[1] != num_classes_local:
            raise ValueError(f"[EMPO3] MLP probs dim {mlp_probs.shape[1]} != num_classes {num_classes_local}")

        w_fusion, w_mlp = empo_cfg.get("ensemble_weights", (0.7, 0.3))
        ensemble_probs = w_fusion * fusion_probs + w_mlp * mlp_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        acc = accuracy_score(y_true, ensemble_preds)
        f1m = f1_score(y_true, ensemble_preds, average="macro")
        prec_m = precision_score(y_true, ensemble_preds, average="macro")
        rec_m  = recall_score(y_true, ensemble_preds, average="macro")

        classes = np.arange(num_classes_local)
        y_true_bin = label_binarize(y_true, classes=classes)
        auroc = roc_auc_score(
            y_true_bin,
            ensemble_probs,
            average="macro",
            multi_class="ovr"
        )

        print(f"\n[EMPO500-ENSEMBLE]  Accuracy:        {acc:.4f}")
        print(f"[EMPO500-ENSEMBLE]  Macro-F1:       {f1m:.4f}")
        print(f"[EMPO500-ENSEMBLE]  Macro-Precision:{prec_m:.4f}")
        print(f"[EMPO500-ENSEMBLE]  Macro-Recall:   {rec_m:.4f}")
        print(f"[EMPO500-ENSEMBLE]  Macro-AUROC:    {auroc:.4f}")

        print("\n=== Ensemble Classification Report ===")
        print(classification_report(y_true, ensemble_preds))


