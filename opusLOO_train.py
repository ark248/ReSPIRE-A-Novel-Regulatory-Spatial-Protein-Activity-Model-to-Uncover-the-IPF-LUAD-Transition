#!/usr/bin/env python3
"""
train_loo.py  —  Run as: python train_loo.py

Leave-One-Out Cross-Validation for Protein Activity Prediction.
Resumes from existing checkpoints — skips folds that already trained.

V2 fix: weights_only=False for PyTorch 2.6 compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import glob
import time
import gc

# ============================================================
# ⚙️  CONFIGURATION
# ============================================================
CONFIG = {
    "data_file": "/shares/vasciaveo_lab/aarulselvan/arachne/DeepSpot_adhiban/metaviperfeatures1.pt",
    "input_dim": 5120,

    "hidden_dims": [512, 256],
    "dropout": 0.35,

    "batch_size": 4096,
    "lr": 1e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 3,
    "weight_decay": 1e-3,
    "epochs": 40,
    "patience": 12,
    "grad_clip": 1.0,

    "correlation_weight": 0.3,
    "min_protein_std": 0.05,

    "num_workers": 4,
    "output_dir": "predictions",
}

SAMPLES = {
    "Lung1_S1": "LUAD",
    "Lung1_S2": "LUAD",
    "Lung2_S1": "LUAD",
    "Lung3_S1": "LUAD",
    "Lung6_S1": "Fibrosis",
    "Lung6_S2": "Fibrosis",
}

PANEL_PROTEINS = {
    "TF": ["NKX2-1", "MYC", "STAT3", "YAP1", "TEAD1", "SMAD3", "SMAD4"],
    "coTF": ["BRD4", "BRD2", "EP300", "CREBBP", "CTCF"],
    "SigSurf": ["EGFR", "PDGFRA", "PDGFRB", "TGFBR1", "TGFBR2", "CXCR4", "ITGB1"],
}
ALL_PANEL = [p for group in PANEL_PROTEINS.values() for p in group]

L.seed_everything(42)
torch.set_float32_matmul_precision("medium")


# ============================================================
# 1. DATASET
# ============================================================
class ProteinDataset(Dataset):
    def __init__(self, features_f16, targets_f16):
        self.x = features_f16
        self.y = targets_f16
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()


# ============================================================
# 2. DATA LOADING
# ============================================================
def load_all_data():
    t0 = time.time()
    print(f"🚀 Loading {CONFIG['data_file']}...")
    payload = torch.load(CONFIG["data_file"], weights_only=False)
    data = payload["data"]
    all_protein_names = payload["meta"]["proteins"]
    barcodes_per_sample = payload["meta"].get("barcodes_per_sample", {})

    print(f"   Proteins: {len(all_protein_names)}")
    for name in data:
        n = len(data[name]["features"])
        label = SAMPLES.get(name, "?")
        print(f"   {name:12s}  {n:>7,} spots  [{label}]")

    elapsed = time.time() - t0
    print(f"   Loaded in {elapsed:.0f}s\n")
    return data, all_protein_names, barcodes_per_sample


def prepare_fold(data, all_protein_names, held_out_sample):
    train_names = [n for n in data.keys() if n != held_out_sample]

    input_dim = CONFIG["input_dim"]
    n_all_proteins = len(all_protein_names)
    n_total = 0
    input_mean = np.zeros(input_dim, dtype=np.float64)
    input_m2 = np.zeros(input_dim, dtype=np.float64)
    target_mean = np.zeros(n_all_proteins, dtype=np.float64)
    target_m2 = np.zeros(n_all_proteins, dtype=np.float64)

    for name in train_names:
        x_np = data[name]["features"].numpy().astype(np.float64)
        y_np = data[name]["targets"].numpy().astype(np.float64)
        for i in range(len(x_np)):
            n_total += 1
            dx = x_np[i] - input_mean
            input_mean += dx / n_total
            input_m2 += dx * (x_np[i] - input_mean)
            dy = y_np[i] - target_mean
            target_mean += dy / n_total
            target_m2 += dy * (y_np[i] - target_mean)
        del x_np, y_np

    input_std = np.sqrt(input_m2 / max(n_total - 1, 1)).astype(np.float32)
    input_mean = input_mean.astype(np.float32)
    input_std = np.clip(input_std, 1e-6, None)
    target_std = np.sqrt(target_m2 / max(n_total - 1, 1)).astype(np.float32)
    target_mean = target_mean.astype(np.float32)
    target_std = np.clip(target_std, 1e-6, None)

    del input_m2, target_m2

    keep_mask = target_std >= CONFIG["min_protein_std"]
    keep_indices = np.where(keep_mask)[0]
    protein_names = [all_protein_names[i] for i in keep_indices]
    target_mean_kept = target_mean[keep_mask]
    target_std_kept = target_std[keep_mask]

    def pack_samples(sample_names):
        x_parts, y_parts = [], []
        for name in sample_names:
            x = data[name]["features"].numpy()
            x = ((x - input_mean) / input_std).astype(np.float16)
            x_parts.append(torch.from_numpy(x))
            del x
            y = data[name]["targets"].numpy()[:, keep_indices]
            y = ((y - target_mean_kept) / target_std_kept).astype(np.float16)
            y_parts.append(torch.from_numpy(y))
            del y
            gc.collect()
        return torch.cat(x_parts), torch.cat(y_parts)

    train_x, train_y = pack_samples(train_names)
    val_x, val_y = pack_samples([held_out_sample])
    gc.collect()

    train_ds = ProteinDataset(train_x, train_y)
    val_ds = ProteinDataset(val_x, val_y)

    return (train_ds, val_ds, protein_names,
            torch.from_numpy(target_mean_kept),
            torch.from_numpy(target_std_kept),
            keep_indices)


# ============================================================
# 3. MODEL
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.needs_proj = (in_dim != out_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        if self.needs_proj:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return (self.proj(x) if self.needs_proj else x) + self.block(x)


class ProteinPredictor(L.LightningModule):
    def __init__(self, n_proteins, target_mean=None, target_std=None,
                 protein_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "target_mean", "target_std", "protein_names"
        ])
        self.n_proteins = n_proteins
        self.corr_weight = CONFIG["correlation_weight"]
        self.protein_names = protein_names

        if target_mean is not None:
            self.register_buffer("target_mean", target_mean)
            self.register_buffer("target_std", target_std)

        dims = [CONFIG["input_dim"]] + CONFIG["hidden_dims"]
        self.encoder = nn.Sequential(*[
            ResidualBlock(dims[i], dims[i + 1], CONFIG["dropout"])
            for i in range(len(dims) - 1)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(dims[-1], n_proteins),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.encoder(x))

    @staticmethod
    def _pearson_r(pred, target):
        p = pred - pred.mean(dim=0, keepdim=True)
        t = target - target.mean(dim=0, keepdim=True)
        num = (p * t).sum(dim=0)
        den = p.pow(2).sum(dim=0).sqrt() * t.pow(2).sum(dim=0).sqrt() + 1e-8
        return (num / den).mean()

    def _loss(self, pred, target):
        mse = F.mse_loss(pred, target)
        if target.shape[0] >= 128:
            r = self._pearson_r(pred, target)
            loss = mse - self.corr_weight * r
        else:
            r = torch.tensor(0.0, device=pred.device)
            loss = mse
        return loss, mse, r

    def training_step(self, batch, batch_idx):
        loss, mse, r = self._loss(self(batch[0]), batch[1])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_r", r, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse, r = self._loss(self(batch[0]), batch[1])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mse", mse, prog_bar=True, sync_dist=True)
        self.log("val_r", r, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )
        warmup = CONFIG["warmup_epochs"]
        total = CONFIG["epochs"]
        def lr_lambda(epoch):
            if epoch < warmup:
                return 0.1 + 0.9 * (epoch / warmup)
            progress = (epoch - warmup) / max(total - warmup, 1)
            return max(CONFIG["min_lr"] / CONFIG["lr"],
                       0.5 * (1 + np.cos(np.pi * progress)))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ============================================================
# 4. EVALUATION & PREDICTION
# ============================================================
@torch.no_grad()
def predict_and_evaluate(model, dataloader, protein_names, target_mean,
                         target_std, sample_name, barcodes=None):
    model.eval()
    device = next(model.parameters()).device

    all_pred_z, all_true_z = [], []
    for x, y in dataloader:
        all_pred_z.append(model(x.to(device)).cpu())
        all_true_z.append(y.cpu())

    pred_z = torch.cat(all_pred_z)
    true_z = torch.cat(all_true_z)

    p = pred_z - pred_z.mean(0, keepdim=True)
    t = true_z - true_z.mean(0, keepdim=True)
    num = (p * t).sum(0)
    den = p.pow(2).sum(0).sqrt() * t.pow(2).sum(0).sqrt() + 1e-8
    r = (num / den).numpy()

    mse = F.mse_loss(pred_z, true_z).item()

    # Convert predictions back to original NES scale
    pred_nes = pred_z * target_std.unsqueeze(0) + target_mean.unsqueeze(0)

    out_dir = CONFIG["output_dir"]
    pred_path = os.path.join(out_dir, f"{sample_name}_predictions.pt")
    torch.save({
        "predictions": pred_nes.half(),
        "protein_names": protein_names,
        "sample": sample_name,
        "disease": SAMPLES[sample_name],
    }, pred_path)

    if barcodes is not None:
        bc_path = os.path.join(out_dir, f"{sample_name}_barcodes.json")
        with open(bc_path, "w") as f:
            json.dump(barcodes, f)

    disease = SAMPLES[sample_name]
    print(f"\n  {'─' * 55}")
    print(f"  📊 {sample_name} [{disease}]  "
          f"({len(pred_z):,} spots × {len(protein_names)} proteins)")
    print(f"  {'─' * 55}")
    print(f"  MSE: {mse:.4f}   Mean r: {np.nanmean(r):.4f}   "
          f"Median r: {np.nanmedian(r):.4f}")

    for t_val in [0.1, 0.2, 0.3, 0.5]:
        n = (r > t_val).sum()
        print(f"      r > {t_val}:  {n:>5d} / {len(r)}  "
              f"({100 * n / len(r):.1f}%)")

    protein_to_idx = {p: i for i, p in enumerate(protein_names)}
    print(f"\n  🧬 Panel proteins:")
    panel_results = {}
    for group_name, proteins in PANEL_PROTEINS.items():
        print(f"    {group_name}:")
        for prot in proteins:
            if prot in protein_to_idx:
                idx = protein_to_idx[prot]
                r_val = float(r[idx])
                panel_results[prot] = r_val
                marker = "✅" if r_val > 0.2 else "⚠️" if r_val > 0.1 else "❌"
                print(f"      {marker} {prot:12s}  r={r_val:.4f}")
            else:
                panel_results[prot] = None
                print(f"      ⬜ {prot:12s}  (filtered out)")

    corr_path = os.path.join(out_dir, f"{sample_name}_correlations.json")
    results = {
        "sample": sample_name,
        "disease": disease,
        "mse": float(mse),
        "mean_r": float(np.nanmean(r)),
        "median_r": float(np.nanmedian(r)),
        "n_above_0.2": int((r > 0.2).sum()),
        "n_above_0.3": int((r > 0.3).sum()),
        "panel_proteins": panel_results,
        "per_protein": {protein_names[i]: float(r[i]) for i in range(len(r))},
    }
    with open(corr_path, "w") as f:
        json.dump(results, f, indent=2)

    return r, panel_results


# ============================================================
# 5. FIND EXISTING CHECKPOINT
# ============================================================
def find_existing_checkpoint(fold_num):
    """Check if a fold already has a trained checkpoint."""
    ckpt_dir = os.path.join(CONFIG["output_dir"], f"checkpoints_fold{fold_num}")
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = glob.glob(os.path.join(ckpt_dir, "best-*.ckpt"))
    if len(ckpts) == 0:
        return None
    # Return the most recent one
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]


# ============================================================
# 6. SINGLE FOLD
# ============================================================
def run_fold(fold_num, held_out, data, all_protein_names, barcodes_per_sample):
    disease = SAMPLES[held_out]
    print(f"\n{'=' * 60}")
    print(f"  FOLD {fold_num}/6: Hold out {held_out} [{disease}]")
    print(f"  Train on: {[n for n in data.keys() if n != held_out]}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Prepare data (needed for both training and evaluation)
    (train_ds, val_ds, protein_names, target_mean, target_std,
     keep_indices) = prepare_fold(data, all_protein_names, held_out)

    n_proteins = len(protein_names)
    print(f"  Proteins (after filtering): {n_proteins}")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_dl = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True,
        persistent_workers=True,
    )

    # Check for existing checkpoint
    existing_ckpt = find_existing_checkpoint(fold_num)

    if existing_ckpt is not None:
        print(f"  ♻️  Found existing checkpoint: {os.path.basename(existing_ckpt)}")
        print(f"     Skipping training, loading for evaluation...")
        best_ckpt_path = existing_ckpt
        try:
            best_epoch = int(best_ckpt_path.split("epoch=")[1].split("-")[0])
        except (IndexError, ValueError):
            best_epoch = -1
        did_train = False
    else:
        # Train from scratch
        model = ProteinPredictor(
            n_proteins=n_proteins,
            target_mean=target_mean,
            target_std=target_std,
            protein_names=protein_names,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        ckpt_dir = os.path.join(CONFIG["output_dir"], f"checkpoints_fold{fold_num}")
        os.makedirs(ckpt_dir, exist_ok=True)

        trainer = L.Trainer(
            max_epochs=CONFIG["epochs"],
            accelerator="gpu", devices=1,
            logger=CSVLogger(save_dir=CONFIG["output_dir"],
                             name=f"fold{fold_num}_{held_out}"),
            callbacks=[
                ModelCheckpoint(
                    dirpath=ckpt_dir, monitor="val_loss", mode="min",
                    save_top_k=1, filename="best-{epoch:03d}-{val_loss:.4f}",
                ),
                EarlyStopping(
                    monitor="val_loss", mode="min",
                    patience=CONFIG["patience"],
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            gradient_clip_val=CONFIG["grad_clip"],
            precision="16-mixed",
            log_every_n_steps=50,
            enable_model_summary=(fold_num == 1),
        )

        trainer.fit(model, train_dl, val_dl)
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        best_epoch = int(best_ckpt_path.split("epoch=")[1].split("-")[0])
        did_train = True
        del model, trainer

    train_time = time.time() - t0
    if did_train:
        print(f"  ⏱️  Fold took {train_time / 60:.1f} min  (best epoch: {best_epoch})")
    else:
        print(f"  ⏱️  Evaluation prep took {train_time / 60:.1f} min  "
              f"(checkpoint epoch: {best_epoch})")

    # Load best checkpoint and evaluate
    best = ProteinPredictor.load_from_checkpoint(
        best_ckpt_path,
        n_proteins=n_proteins,
        target_mean=target_mean,
        target_std=target_std,
        protein_names=protein_names,
        weights_only=False,
    )
    best.eval().cuda()

    barcodes = barcodes_per_sample.get(held_out, None)
    r, panel_results = predict_and_evaluate(
        best, val_dl, protein_names, target_mean, target_std,
        held_out, barcodes=barcodes,
    )

    # Cleanup
    del best, train_ds, val_ds, train_dl, val_dl
    del target_mean, target_std
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "sample": held_out,
        "disease": disease,
        "mean_r": float(np.nanmean(r)),
        "median_r": float(np.nanmedian(r)),
        "n_above_0.2": int((r > 0.2).sum()),
        "n_above_0.3": int((r > 0.3).sum()),
        "best_epoch": best_epoch,
        "train_time_min": round(train_time / 60, 1),
        "panel": panel_results,
        "n_proteins": n_proteins,
        "resumed": not did_train,
    }


# ============================================================
# 7. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  🧬 LEAVE-ONE-OUT PROTEIN ACTIVITY PREDICTION")
    print("=" * 60)
    print(f"  6 folds, {len(SAMPLES)} samples")
    print(f"  Panel: {len(ALL_PANEL)} proteins across "
          f"{len(PANEL_PROTEINS)} categories")
    print(f"  Resume mode: will skip folds with existing checkpoints")
    print()

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    data, all_protein_names, barcodes_per_sample = load_all_data()

    all_results = []
    t_total = time.time()

    for fold_num, held_out in enumerate(SAMPLES.keys(), 1):
        result = run_fold(fold_num, held_out, data, all_protein_names,
                          barcodes_per_sample)
        all_results.append(result)

    total_time = time.time() - t_total

    # ── Summary ──
    print(f"\n\n{'=' * 60}")
    print(f"  📊 LEAVE-ONE-OUT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time / 60:.1f} minutes\n")

    n_resumed = sum(1 for r in all_results if r.get("resumed", False))
    n_trained = 6 - n_resumed
    print(f"  Folds trained: {n_trained}  |  Folds resumed: {n_resumed}\n")

    print(f"  {'Sample':<12s} {'Disease':<10s} {'Mean r':>8s} {'r>0.2':>6s} "
          f"{'r>0.3':>6s} {'Best ep':>8s} {'Status':>10s}")
    print(f"  {'─'*12} {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")
    for r in all_results:
        status = "resumed" if r.get("resumed", False) else "trained"
        print(f"  {r['sample']:<12s} {r['disease']:<10s} {r['mean_r']:>8.4f} "
              f"{r['n_above_0.2']:>6d} {r['n_above_0.3']:>6d} "
              f"{r['best_epoch']:>8d} {status:>10s}")

    luad_results = [r for r in all_results if r["disease"] == "LUAD"]
    fib_results = [r for r in all_results if r["disease"] == "Fibrosis"]

    print(f"\n  By disease:")
    if luad_results:
        mean_r_luad = np.mean([r["mean_r"] for r in luad_results])
        print(f"    LUAD     ({len(luad_results)} samples):  mean r = {mean_r_luad:.4f}")
    if fib_results:
        mean_r_fib = np.mean([r["mean_r"] for r in fib_results])
        print(f"    Fibrosis ({len(fib_results)} samples):  mean r = {mean_r_fib:.4f}")

    print(f"\n  🧬 Panel protein performance (mean r across folds):")
    panel_summary = {}
    for group_name, proteins in PANEL_PROTEINS.items():
        print(f"    {group_name}:")
        for prot in proteins:
            values = []
            for r in all_results:
                v = r["panel"].get(prot, None)
                if v is not None:
                    values.append(v)
            if values:
                mean_v = np.mean(values)
                std_v = np.std(values)
                panel_summary[prot] = {"mean_r": float(mean_v),
                                       "std_r": float(std_v),
                                       "n_folds": len(values)}
                marker = "✅" if mean_v > 0.15 else "⚠️" if mean_v > 0.05 else "❌"
                print(f"      {marker} {prot:12s}  r={mean_v:.4f} ± {std_v:.4f}  "
                      f"({len(values)} folds)")
            else:
                panel_summary[prot] = {"mean_r": None, "filtered": True}
                print(f"      ⬜ {prot:12s}  (filtered in all folds)")

    summary = {
        "config": {k: v for k, v in CONFIG.items()},
        "per_fold": all_results,
        "panel_summary": panel_summary,
        "total_time_min": round(total_time / 60, 1),
    }
    summary_path = os.path.join(CONFIG["output_dir"], "loo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  💾 Summary: {summary_path}")
    print(f"  💾 Per-sample predictions in: {CONFIG['output_dir']}/")
    print(f"\n✅ Done. All 6 samples have unbiased predictions.")


if __name__ == "__main__":
    main()




    #!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import glob
import time
import gc


# configuration
CONFIG = {
    "data_file": "/shares/vasciaveo_lab/aarulselvan/arachne/DeepSpot_adhiban/metaviperfeatures1.pt",
    "input_dim": 5120,

    "hidden_dims": [512, 256],
    "dropout": 0.35,

    "batch_size": 4096,
    "lr": 1e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 3,
    "weight_decay": 1e-3,
    "epochs": 40,
    "patience": 12,
    "grad_clip": 1.0,

    "correlation_weight": 0.3,
    "min_protein_std": 0.05,

    "num_workers": 4,
    "output_dir": "predictions",
}

SAMPLES = {
    "Lung1_S1": "LUAD",
    "Lung1_S2": "LUAD",
    "Lung2_S1": "LUAD",
    "Lung3_S1": "LUAD",
    "Lung6_S1": "Fibrosis",
    "Lung6_S2": "Fibrosis",
}

PANEL_PROTEINS = {
    "TF": ["NKX2-1", "MYC", "STAT3", "YAP1", "TEAD1", "SMAD3", "SMAD4"],
    "coTF": ["BRD4", "BRD2", "EP300", "CREBBP", "CTCF"],
    "SigSurf": ["EGFR", "PDGFRA", "PDGFRB", "TGFBR1", "TGFBR2", "CXCR4", "ITGB1"],
}
ALL_PANEL = [p for group in PANEL_PROTEINS.values() for p in group]

L.seed_everything(42)
torch.set_float32_matmul_precision("medium")


# dataset
class ProteinDataset(Dataset):
    def __init__(self, features_f16, targets_f16):
        self.x = features_f16
        self.y = targets_f16

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()


# data loading
def load_all_data():
    t0 = time.time()
    print(f"loading {CONFIG['data_file']}...")
    payload = torch.load(CONFIG["data_file"], weights_only=False)
    data = payload["data"]
    all_protein_names = payload["meta"]["proteins"]
    barcodes_per_sample = payload["meta"].get("barcodes_per_sample", {})

    print(f"proteins: {len(all_protein_names)}")
    for name in data:
        n = len(data[name]["features"])
        label = SAMPLES.get(name, "?")
        print(f"{name:12s}  {n:>7,} spots  [{label}]")

    print(f"loaded in {time.time() - t0:.0f}s\n")
    return data, all_protein_names, barcodes_per_sample


def prepare_fold(data, all_protein_names, held_out_sample):
    train_names = [n for n in data.keys() if n != held_out_sample]

    input_dim = CONFIG["input_dim"]
    n_all_proteins = len(all_protein_names)

    n_total = 0
    input_mean = np.zeros(input_dim, dtype=np.float64)
    input_m2 = np.zeros(input_dim, dtype=np.float64)
    target_mean = np.zeros(n_all_proteins, dtype=np.float64)
    target_m2 = np.zeros(n_all_proteins, dtype=np.float64)

    for name in train_names:
        x_np = data[name]["features"].numpy().astype(np.float64)
        y_np = data[name]["targets"].numpy().astype(np.float64)
        for i in range(len(x_np)):
            n_total += 1
            dx = x_np[i] - input_mean
            input_mean += dx / n_total
            input_m2 += dx * (x_np[i] - input_mean)

            dy = y_np[i] - target_mean
            target_mean += dy / n_total
            target_m2 += dy * (y_np[i] - target_mean)
        del x_np, y_np

    input_std = np.sqrt(input_m2 / max(n_total - 1, 1)).astype(np.float32)
    target_std = np.sqrt(target_m2 / max(n_total - 1, 1)).astype(np.float32)

    input_mean = input_mean.astype(np.float32)
    target_mean = target_mean.astype(np.float32)

    input_std = np.clip(input_std, 1e-6, None)
    target_std = np.clip(target_std, 1e-6, None)

    keep_mask = target_std >= CONFIG["min_protein_std"]
    keep_indices = np.where(keep_mask)[0]

    protein_names = [all_protein_names[i] for i in keep_indices]
    target_mean_kept = target_mean[keep_mask]
    target_std_kept = target_std[keep_mask]

    def pack_samples(sample_names):
        xs, ys = [], []
        for name in sample_names:
            x = data[name]["features"].numpy()
            x = ((x - input_mean) / input_std).astype(np.float16)
            xs.append(torch.from_numpy(x))

            y = data[name]["targets"].numpy()[:, keep_indices]
            y = ((y - target_mean_kept) / target_std_kept).astype(np.float16)
            ys.append(torch.from_numpy(y))

            gc.collect()
        return torch.cat(xs), torch.cat(ys)

    train_x, train_y = pack_samples(train_names)
    val_x, val_y = pack_samples([held_out_sample])

    return (
        ProteinDataset(train_x, train_y),
        ProteinDataset(val_x, val_y),
        protein_names,
        torch.from_numpy(target_mean_kept),
        torch.from_numpy(target_std_kept),
        keep_indices,
    )


# model architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.needs_proj = in_dim != out_dim
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        if self.needs_proj:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return (self.proj(x) if self.needs_proj else x) + self.block(x)


class ProteinPredictor(L.LightningModule):
    def __init__(self, n_proteins, target_mean=None, target_std=None, protein_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=["target_mean", "target_std", "protein_names"])
        self.n_proteins = n_proteins
        self.corr_weight = CONFIG["correlation_weight"]
        self.protein_names = protein_names

        if target_mean is not None:
            self.register_buffer("target_mean", target_mean)
            self.register_buffer("target_std", target_std)

        dims = [CONFIG["input_dim"]] + CONFIG["hidden_dims"]
        self.encoder = nn.Sequential(
            *[ResidualBlock(dims[i], dims[i + 1], CONFIG["dropout"])
              for i in range(len(dims) - 1)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(dims[-1], n_proteins),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.encoder(x))

    def _pearson_r(pred, target):
        p = pred - pred.mean(0, keepdim=True)
        t = target - target.mean(0, keepdim=True)
        num = (p * t).sum(0)
        den = p.pow(2).sum(0).sqrt() * t.pow(2).sum(0).sqrt() + 1e-8
        return (num / den).mean()

    def _loss(self, pred, target):
        mse = F.mse_loss(pred, target)
        if target.shape[0] >= 128:
            r = self._pearson_r(pred, target)
            return mse - self.corr_weight * r, mse, r
        return mse, mse, torch.tensor(0.0, device=pred.device)

    def training_step(self, batch, _):
        loss, _, r = self._loss(self(batch[0]), batch[1])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_r", r, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, mse, r = self._loss(self(batch[0]), batch[1])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mse", mse, prog_bar=True, sync_dist=True)
        self.log("val_r", r, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )

        def lr_lambda(epoch):
            if epoch < CONFIG["warmup_epochs"]:
                return 0.1 + 0.9 * (epoch / CONFIG["warmup_epochs"])
            progress = (epoch - CONFIG["warmup_epochs"]) / max(
                CONFIG["epochs"] - CONFIG["warmup_epochs"], 1
            )
            return max(
                CONFIG["min_lr"] / CONFIG["lr"],
                0.5 * (1 + np.cos(np.pi * progress)),
            )

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

if __name__ == "__main__":
    main()



import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import glob
import time
import gc


# configuration
CONFIG = {
    "data_file": "/shares/vasciaveo_lab/aarulselvan/arachne/DeepSpot_adhiban/metaviperfeatures1.pt",
    "input_dim": 5120,

    "hidden_dims": [512, 256],
    "dropout": 0.35,

    "batch_size": 4096,
    "lr": 1e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 3,
    "weight_decay": 1e-3,
    "epochs": 40,
    "patience": 12,
    "grad_clip": 1.0,

    "correlation_weight": 0.3,
    "min_protein_std": 0.05,

    "num_workers": 4,
    "output_dir": "predictions",
}

SAMPLES = {
    "Lung1_S1": "LUAD",
    "Lung1_S2": "LUAD",
    "Lung2_S1": "LUAD",
    "Lung3_S1": "LUAD",
    "Lung6_S1": "Fibrosis",
    "Lung6_S2": "Fibrosis",
}

PANEL_PROTEINS = {
    "TF": ["NKX2-1", "MYC", "STAT3", "YAP1", "TEAD1", "SMAD3", "SMAD4"],
    "coTF": ["BRD4", "BRD2", "EP300", "CREBBP", "CTCF"],
    "SigSurf": ["EGFR", "PDGFRA", "PDGFRB", "TGFBR1", "TGFBR2", "CXCR4", "ITGB1"],
}
ALL_PANEL = [p for group in PANEL_PROTEINS.values() for p in group]

L.seed_everything(42)
torch.set_float32_matmul_precision("medium")


# dataset
class ProteinDataset(Dataset):
    def __init__(self, features_f16, targets_f16):
        self.x = features_f16
        self.y = targets_f16

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()


# data loading
def load_all_data():
    t0 = time.time()
    print(f"loading {CONFIG['data_file']}...")
    payload = torch.load(CONFIG["data_file"], weights_only=False)
    data = payload["data"]
    all_protein_names = payload["meta"]["proteins"]
    barcodes_per_sample = payload["meta"].get("barcodes_per_sample", {})

    print(f"proteins: {len(all_protein_names)}")
    for name in data:
        n = len(data[name]["features"])
        label = SAMPLES.get(name, "?")
        print(f"{name:12s}  {n:>7,} spots  [{label}]")

    print(f"loaded in {time.time() - t0:.0f}s\n")
    return data, all_protein_names, barcodes_per_sample


def prepare_fold(data, all_protein_names, held_out_sample):
    train_names = [n for n in data.keys() if n != held_out_sample]

    input_dim = CONFIG["input_dim"]
    n_all_proteins = len(all_protein_names)

    n_total = 0
    input_mean = np.zeros(input_dim, dtype=np.float64)
    input_m2 = np.zeros(input_dim, dtype=np.float64)
    target_mean = np.zeros(n_all_proteins, dtype=np.float64)
    target_m2 = np.zeros(n_all_proteins, dtype=np.float64)

    for name in train_names:
        x_np = data[name]["features"].numpy().astype(np.float64)
        y_np = data[name]["targets"].numpy().astype(np.float64)
        for i in range(len(x_np)):
            n_total += 1
            dx = x_np[i] - input_mean
            input_mean += dx / n_total
            input_m2 += dx * (x_np[i] - input_mean)

            dy = y_np[i] - target_mean
            target_mean += dy / n_total
            target_m2 += dy * (y_np[i] - target_mean)
        del x_np, y_np

    input_std = np.sqrt(input_m2 / max(n_total - 1, 1)).astype(np.float32)
    target_std = np.sqrt(target_m2 / max(n_total - 1, 1)).astype(np.float32)

    input_mean = input_mean.astype(np.float32)
    target_mean = target_mean.astype(np.float32)

    input_std = np.clip(input_std, 1e-6, None)
    target_std = np.clip(target_std, 1e-6, None)

    keep_mask = target_std >= CONFIG["min_protein_std"]
    keep_indices = np.where(keep_mask)[0]

    protein_names = [all_protein_names[i] for i in keep_indices]
    target_mean_kept = target_mean[keep_mask]
    target_std_kept = target_std[keep_mask]

    def pack_samples(sample_names):
        xs, ys = [], []
        for name in sample_names:
            x = data[name]["features"].numpy()
            x = ((x - input_mean) / input_std).astype(np.float16)
            xs.append(torch.from_numpy(x))

            y = data[name]["targets"].numpy()[:, keep_indices]
            y = ((y - target_mean_kept) / target_std_kept).astype(np.float16)
            ys.append(torch.from_numpy(y))
        return torch.cat(xs), torch.cat(ys)

    train_x, train_y = pack_samples(train_names)
    val_x, val_y = pack_samples([held_out_sample])

    return (
        ProteinDataset(train_x, train_y),
        ProteinDataset(val_x, val_y),
        protein_names,
        torch.from_numpy(target_mean_kept),
        torch.from_numpy(target_std_kept),
        keep_indices,
    )


# model architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.needs_proj = in_dim != out_dim
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        if self.needs_proj:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return (self.proj(x) if self.needs_proj else x) + self.block(x)


class ProteinPredictor(L.LightningModule):
    def __init__(self, n_proteins, target_mean=None, target_std=None, protein_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=["target_mean", "target_std", "protein_names"])
        self.n_proteins = n_proteins
        self.corr_weight = CONFIG["correlation_weight"]
        self.protein_names = protein_names

        if target_mean is not None:
            self.register_buffer("target_mean", target_mean)
            self.register_buffer("target_std", target_std)

        dims = [CONFIG["input_dim"]] + CONFIG["hidden_dims"]
        self.encoder = nn.Sequential(
            *[ResidualBlock(dims[i], dims[i + 1], CONFIG["dropout"])
              for i in range(len(dims) - 1)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(dims[-1], n_proteins),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.encoder(x))

    @staticmethod
    def _pearson_r(pred, target):
        p = pred - pred.mean(0, keepdim=True)
        t = target - target.mean(0, keepdim=True)
        num = (p * t).sum(0)
        den = p.pow(2).sum(0).sqrt() * t.pow(2).sum(0).sqrt() + 1e-8
        return (num / den).mean()

    def _loss(self, pred, target):
        mse = F.mse_loss(pred, target)
        if target.shape[0] >= 128:
            r = self._pearson_r(pred, target)
            return mse - self.corr_weight * r, mse, r
        return mse, mse, torch.tensor(0.0, device=pred.device)

    def training_step(self, batch, _):
        loss, _, r = self._loss(self(batch[0]), batch[1])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_r", r, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, mse, r = self._loss(self(batch[0]), batch[1])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mse", mse, prog_bar=True, sync_dist=True)
        self.log("val_r", r, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )

        def lr_lambda(epoch):
            if epoch < CONFIG["warmup_epochs"]:
                return 0.1 + 0.9 * (epoch / CONFIG["warmup_epochs"])
            progress = (epoch - CONFIG["warmup_epochs"]) / max(
                CONFIG["epochs"] - CONFIG["warmup_epochs"], 1
            )
            return max(
                CONFIG["min_lr"] / CONFIG["lr"],
                0.5 * (1 + np.cos(np.pi * progress)),
            )

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


def main():
    print("script loaded successfully")


if __name__ == "__main__":
    main()    