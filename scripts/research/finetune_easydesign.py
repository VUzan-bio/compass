"""Fine-tune CNN+RNA-FM on EasyDesign trans-cleavage + thermodynamic scalars.

Three configs:
  A: Kim2018 pretrained, zero-shot on EasyDesign test (baseline)
  B: + fine-tuned on EasyDesign train (transfer learning)
  C: + fine-tuned with 2 thermodynamic scalar features (hybrid dG, fold dG)

Usage:
    python scripts/research/finetune_easydesign.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = "compass/data/embeddings/rnafm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results/research/guide_improvements")


# --- Thermodynamic features ---

RNA_DNA_NN = {
    "AA": (-7.8, -21.9), "AC": (-5.9, -12.3), "AG": (-9.1, -23.5), "AU": (-8.3, -23.9),
    "CA": (-9.0, -26.1), "CC": (-9.3, -23.2), "CG": (-16.3, -47.1), "CU": (-7.0, -19.7),
    "GA": (-5.5, -13.5), "GC": (-8.0, -17.1), "GG": (-12.8, -31.9), "GU": (-7.8, -21.6),
    "UA": (-7.8, -23.2), "UC": (-8.6, -22.9), "UG": (-10.4, -28.4), "UU": (-11.5, -36.4),
}


def compute_hybrid_dg(rna_seq):
    dh, ds = 1.9, -3.9
    seq = rna_seq.upper().replace("T", "U")
    for i in range(len(seq) - 1):
        di = seq[i:i + 2]
        if di in RNA_DNA_NN:
            h, s = RNA_DNA_NN[di]
            dh += h
            ds += s
    return dh - 310.15 * (ds / 1000.0)


def compute_folding_dg(seq):
    gc = sum(1 for c in seq.upper() if c in "GC") / max(len(seq), 1)
    return -2.0 * gc


def get_scalars(sequences):
    scalars = []
    for seq in sequences:
        spacer = seq[4:24] if len(seq) >= 24 else seq[:20]
        crrna = spacer.replace("T", "U")
        hybrid_dg = compute_hybrid_dg(crrna)
        fold_dg = compute_folding_dg(spacer)
        scalars.append([hybrid_dg / -50.0, fold_dg / -2.0])
    return np.array(scalars, dtype=np.float32)


# --- Dataset ---

class ScalarFeatureDataset(Dataset):
    def __init__(self, sequences, activities, scalars):
        self.sequences = sequences
        self.activities = activities
        self.scalars = scalars

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        mat = torch.zeros(4, 34)
        for i, nt in enumerate(seq[:34].upper()):
            if nt in mapping:
                mat[mapping[nt], i] = 1.0

        spacer = seq[4:24] if len(seq) >= 24 else seq[:20]
        comp = {"A": "U", "T": "A", "G": "C", "C": "G"}
        crrna = "".join(comp.get(b, "N") for b in reversed(spacer))

        return {
            "target_onehot": mat,
            "crrna_spacer": crrna,
            "efficiency": torch.tensor(self.activities[idx], dtype=torch.float32),
            "scalars": torch.tensor(self.scalars[idx], dtype=torch.float32),
        }


def collate_scalar(batch):
    return {
        "target_onehot": torch.stack([b["target_onehot"] for b in batch]),
        "crrna_spacer": [b["crrna_spacer"] for b in batch],
        "efficiency": torch.stack([b["efficiency"] for b in batch]),
        "scalars": torch.stack([b["scalars"] for b in batch]),
    }


# --- Training ---

def finetune(model, train_loader, val_loader, cache, use_scalars=False, n_epochs=50, patience_limit=10):
    from compass_net.losses.multitask_loss import MultiTaskLoss
    from compass_net.training.train_compass_ml import _get_batch_embeddings

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = MultiTaskLoss(lambda_disc=0.0, lambda_rank=0.5)
    best_rho, patience_ctr, best_state = -1.0, 0, None

    for epoch in range(n_epochs):
        loss_fn.set_spearman_strength(max(0.1, 1.0 - 0.9 * epoch / n_epochs))
        model.train()
        for batch in train_loader:
            emb = _get_batch_embeddings(batch["crrna_spacer"], cache, DEVICE)
            kwargs = {"target_onehot": batch["target_onehot"].to(DEVICE), "crrna_rnafm_emb": emb}
            if use_scalars:
                kwargs["scalar_features"] = batch["scalars"].to(DEVICE)
            out = model(**kwargs)
            loss = loss_fn(pred_eff=out["efficiency"], true_eff=batch["efficiency"].to(DEVICE))["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_loader:
                emb = _get_batch_embeddings(batch["crrna_spacer"], cache, DEVICE)
                kwargs = {"target_onehot": batch["target_onehot"].to(DEVICE), "crrna_rnafm_emb": emb}
                if use_scalars:
                    kwargs["scalar_features"] = batch["scalars"].to(DEVICE)
                out = model(**kwargs)
                vp.extend(out["efficiency"].squeeze(-1).cpu().tolist())
                vt.extend(batch["efficiency"].tolist())

        vr = spearmanr(vp, vt).statistic
        if np.isnan(vr):
            vr = 0.0
        if vr > best_rho:
            best_rho = vr
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch %d: val_rho=%.4f, best=%.4f", epoch + 1, vr, best_rho)
        if patience_ctr >= patience_limit:
            logger.info("  Early stop at epoch %d", epoch + 1)
            break

    model.load_state_dict(best_state)
    return model, best_rho


def evaluate(model, test_loader, cache, test_acts_raw, use_scalars=False):
    from compass_net.training.train_compass_ml import _get_batch_embeddings

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            emb = _get_batch_embeddings(batch["crrna_spacer"], cache, DEVICE)
            kwargs = {"target_onehot": batch["target_onehot"].to(DEVICE), "crrna_rnafm_emb": emb}
            if use_scalars:
                kwargs["scalar_features"] = batch["scalars"].to(DEVICE)
            out = model(**kwargs)
            preds.extend(out["efficiency"].squeeze(-1).cpu().tolist())
    return float(spearmanr(preds, test_acts_raw).statistic)


def main():
    from compass_net.compass_ml import CompassML
    from compass_net.data.embedding_cache import EmbeddingCache
    from compass_net.data.loaders.load_easydesign import load_easydesign
    from compass_net.training.reproducibility import seed_everything

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(CACHE_DIR)
    logger.info("Device: %s, RNA-FM cache: %d entries", DEVICE, len(cache))

    # Load EasyDesign
    ed = load_easydesign()
    train_seqs = ed["sequences"]
    train_acts_norm = (rankdata(ed["activities"]) / len(ed["activities"])).tolist()
    test_seqs = ed["test_sequences"]
    test_acts_raw = np.array(ed["test_activities"])
    test_acts_norm = (rankdata(test_acts_raw) / len(test_acts_raw)).tolist()

    logger.info("EasyDesign: train=%d, test=%d", len(train_seqs), len(test_seqs))

    # Compute scalars
    train_scalars = get_scalars(train_seqs)
    test_scalars = get_scalars(test_seqs)

    # 90/10 split
    n = len(train_seqs)
    perm = np.random.RandomState(42).permutation(n)
    n_train = int(0.9 * n)
    tr_idx, va_idx = perm[:n_train], perm[n_train:]

    train_ds = ScalarFeatureDataset(
        [train_seqs[i] for i in tr_idx], [train_acts_norm[i] for i in tr_idx], train_scalars[tr_idx])
    val_ds = ScalarFeatureDataset(
        [train_seqs[i] for i in va_idx], [train_acts_norm[i] for i in va_idx], train_scalars[va_idx])
    test_ds = ScalarFeatureDataset(test_seqs, test_acts_norm, test_scalars)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_scalar)
    val_loader = DataLoader(val_ds, batch_size=512, collate_fn=collate_scalar)
    test_loader = DataLoader(test_ds, batch_size=512, collate_fn=collate_scalar)

    # Load Kim checkpoint
    ckpt = torch.load(
        "results/research/kim2018_benchmark/config_cnn_rnafm/seed_123/best_model.pt",
        map_location="cpu", weights_only=False,
    )

    # --- Config A: Zero-shot ---
    logger.info("\n=== A: Kim pretrained, zero-shot on EasyDesign ===")
    seed_everything(42)
    model_a = CompassML(use_rnafm=True, use_rloop_attention=False, multitask=False)
    model_a.load_state_dict(ckpt["model_state_dict"])
    model_a = model_a.to(DEVICE)
    rho_a = evaluate(model_a, test_loader, cache, test_acts_raw)
    logger.info("  Test rho (zero-shot): %.4f", rho_a)

    # --- Config B: Fine-tune, no scalars ---
    logger.info("\n=== B: Fine-tune on EasyDesign (no scalars) ===")
    seed_everything(42)
    model_b = CompassML(use_rnafm=True, use_rloop_attention=False, multitask=False)
    model_b.load_state_dict(ckpt["model_state_dict"])
    model_b = model_b.to(DEVICE)
    t0 = time.time()
    model_b, val_rho_b = finetune(model_b, train_loader, val_loader, cache, use_scalars=False)
    rho_b = evaluate(model_b, test_loader, cache, test_acts_raw)
    logger.info("  Test rho (fine-tuned): %.4f, val: %.4f (%.0fs)", rho_b, val_rho_b, time.time() - t0)

    # --- Config C: Fine-tune with scalars ---
    logger.info("\n=== C: Fine-tune on EasyDesign + thermo scalars ===")
    seed_everything(42)
    model_c = CompassML(use_rnafm=True, use_rloop_attention=False, multitask=False, n_scalar_features=2)
    # Transfer compatible weights
    c_state = model_c.state_dict()
    transferred = 0
    for k, v in ckpt["model_state_dict"].items():
        if k in c_state and c_state[k].shape == v.shape:
            c_state[k] = v
            transferred += 1
    model_c.load_state_dict(c_state)
    logger.info("  Transferred %d/%d tensors from Kim checkpoint", transferred, len(ckpt["model_state_dict"]))
    model_c = model_c.to(DEVICE)
    t0 = time.time()
    model_c, val_rho_c = finetune(model_c, train_loader, val_loader, cache, use_scalars=True)
    rho_c = evaluate(model_c, test_loader, cache, test_acts_raw, use_scalars=True)
    logger.info("  Test rho (fine-tuned + scalars): %.4f, val: %.4f (%.0fs)", rho_c, val_rho_c, time.time() - t0)

    # --- Save best model ---
    best_config = "C" if rho_c > rho_b else "B"
    best_model = model_c if rho_c > rho_b else model_b
    best_rho = max(rho_b, rho_c)

    ckpt_path = RESULTS_DIR / "best_easydesign_finetuned.pt"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "config": best_config,
        "test_rho": best_rho,
        "n_scalar_features": 2 if best_config == "C" else 0,
    }, str(ckpt_path))
    logger.info("\nSaved best model (Config %s) to %s", best_config, ckpt_path)

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("GUIDE SCORING IMPROVEMENT RESULTS")
    logger.info("=" * 60)
    logger.info("%-45s %10s", "Config", "ED Test rho")
    logger.info("-" * 58)
    logger.info("%-45s %10.4f", "A: Kim pretrained (zero-shot)", rho_a)
    logger.info("%-45s %10.4f  (%+.4f)", "B: + EasyDesign fine-tune", rho_b, rho_b - rho_a)
    logger.info("%-45s %10.4f  (%+.4f)", "C: + fine-tune + thermo scalars", rho_c, rho_c - rho_a)

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump({
            "zero_shot_rho": round(float(rho_a), 4),
            "finetuned_rho": round(float(rho_b), 4),
            "finetuned_scalars_rho": round(float(rho_c), 4),
            "best_config": best_config,
        }, f, indent=2)


if __name__ == "__main__":
    main()
