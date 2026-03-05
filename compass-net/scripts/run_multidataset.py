"""Multi-dataset domain-adversarial training for Compass-ML.

Combines Kim 2018 (HT1-1 + HT1-2) with EasyDesign trans-cleavage data.
Uses gradient reversal (Ganin et al. 2016) to learn domain-invariant
features that capture universal Cas12a biology, not batch effects.

Evaluation on the SAME Kim 2018 HT2+HT3 test set for fair comparison
with single-dataset rows.

Usage (from compass/ root):
    python compass-net/scripts/run_multidataset.py --device cuda
"""

from __future__ import annotations

import sys, os, argparse, logging, json, math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _COMPASS_NET_DIR)

from run_phase1 import _setup, load_kim2018_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compass-ML multi-dataset + domain adversarial")
    parser.add_argument("--data", type=str, default="compass/data/kim2018/nbt4061_source_data.xlsx")
    parser.add_argument("--easydesign", type=str,
                        default="compass-net/data/external/easydesign/Table_S2.xlsx")
    parser.add_argument("--cache-dir", type=str, default="E:/compass-net-data/cache/rnafm")
    parser.add_argument("--pretrained", type=str,
                        default="E:/compass-net-data/weights/phase1_rlpa_best.pt")
    parser.add_argument("--output", type=str,
                        default="E:/compass-net-data/weights/multidataset_da_best.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-easydesign", action="store_true",
                        help="Fallback: use Kim 2018 intra-domain split only")
    parser.add_argument("--lambda-domain", type=float, default=0.05,
                        help="Weight for domain adversarial loss (0.05 = gentle)")
    args = parser.parse_args()

    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.multi_dataset import MultiDatasetLoader, DatasetMeta, collate_multi
    from compass_ml.data.balanced_sampler import DomainBalancedSampler
    from compass_ml.data.embedding_cache import EmbeddingCache
    from compass_ml.training.train_compass_ml import _get_batch_embeddings
    from compass_ml.training.reproducibility import seed_everything
    from compass_ml.heads.domain_head import domain_adaptation_lambda
    from torch.utils.data import DataLoader

    seed_everything(args.seed)
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # --- Load Kim 2018 as multi-domain ---
    logger.info("Loading Kim 2018 data (multi-domain)...")
    from compass_ml.data.loaders.load_kim2018 import load_kim2018_domains
    kim_data = load_kim2018_domains(args.data)

    datasets = []
    domain_id = 0
    for d in kim_data["train_domains"]:
        datasets.append({
            "metadata": DatasetMeta(
                name=d["name"], domain_id=domain_id, variant=d["variant"],
                readout_type=d["readout_type"], seq_format=d["seq_format"],
                cell_context=d["cell_context"],
            ),
            "sequences": d["sequences"],
            "activities": d["activities"],
        })
        domain_id += 1

    # --- Load EasyDesign (if available) ---
    easydesign_available = False
    if not args.no_easydesign:
        try:
            from compass_ml.data.loaders.load_easydesign import load_easydesign
            ed_data = load_easydesign(args.easydesign, use_augmented=False)
            datasets.append({
                "metadata": DatasetMeta(
                    name=ed_data["name"], domain_id=domain_id,
                    variant=ed_data["variant"],
                    readout_type=ed_data["readout_type"],
                    seq_format=ed_data["seq_format"],
                    cell_context=ed_data["cell_context"],
                ),
                "sequences": ed_data["sequences"],
                "activities": ed_data["activities"],
            })
            easydesign_available = True
            logger.info("EasyDesign loaded: %d training samples", len(ed_data["sequences"]))
            domain_id += 1
        except FileNotFoundError as e:
            logger.warning("EasyDesign not available: %s", e)
            logger.warning("Falling back to Kim 2018 intra-domain split only")

    n_domains = len(datasets)
    logger.info("Training with %d domains", n_domains)

    # --- Build unified dataset ---
    train_dataset = MultiDatasetLoader(datasets)

    # --- Held-out test set: Kim 2018 HT2+HT3 (same as all previous rows) ---
    (_, _), (seqs_val, y_val), (seqs_test, y_test) = load_kim2018_sequences(args.data)
    logger.info("Test set: %d sequences (Kim 2018 HT2+HT3)", len(seqs_test))

    # --- Embedding cache (for frozen RNA-FM mode) ---
    cache = EmbeddingCache(args.cache_dir)
    logger.info("Embedding cache: %d entries", len(cache))

    # --- Balanced sampler ---
    sampler = DomainBalancedSampler(
        domain_ids=train_dataset.domain_ids,
        batch_size=args.batch_size,
        n_batches_per_epoch=200,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_multi,
    )

    # Val/test loaders (standard, from Kim 2018 data using existing pipeline)
    from compass_ml.data.paired_loader import SingleTargetDataset
    from compass_ml.training.train_compass_ml import collate_single_target

    val_ds = SingleTargetDataset(seqs_val, y_val.tolist())
    test_ds = SingleTargetDataset(seqs_test, y_test.tolist())
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_single_target)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_single_target)

    # --- Build model ---
    model = CompassML(
        use_rnafm=True,
        use_rloop_attention=True,
        multitask=False,
        n_domains=n_domains,
    )

    # Transfer weights from Phase 1 RLPA checkpoint
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    pretrained_state = ckpt["model_state_dict"]
    model_state = model.state_dict()

    transferred = 0
    for name, param in model_state.items():
        if name in pretrained_state and pretrained_state[name].shape == param.shape:
            model_state[name] = pretrained_state[name]
            transferred += 1

    model.load_state_dict(model_state)
    logger.info("Weight transfer: %d layers from %s", transferred, args.pretrained)

    model = model.to(device)
    logger.info("Trainable params: %d", model.count_trainable_params())

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=40, T_mult=2, eta_min=1e-6,
    )

    efficiency_loss_fn = nn.HuberLoss(delta=0.5)
    domain_loss_fn = nn.CrossEntropyLoss()

    # --- Training loop ---
    best_rho = -1.0
    patience_counter = 0
    patience = 20
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("=" * 60)
    logger.info("Multi-dataset domain-adversarial training")
    logger.info("  Domains: %d | Batch: %d | LR: %.1e | lambda_domain: %.3f",
                n_domains, args.batch_size, args.lr, args.lambda_domain)
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        progress = epoch / max(args.epochs - 1, 1)
        grl_lambda = domain_adaptation_lambda(progress)

        # Set GRL lambda for this epoch
        if model.use_domain_adversarial:
            model.domain_head.set_lambda(grl_lambda)

        total_eff_loss = 0.0
        total_dom_loss = 0.0
        total_dom_correct = 0
        total_dom_count = 0
        n_batches = 0

        for batch in train_loader:
            target_oh = batch["target_onehot"].to(device)
            crrna_spacers = batch["crrna_spacers"]
            activity = batch["activity"].to(device)
            domain_ids = batch["domain_id"].to(device)

            # Get RNA-FM embeddings from cache
            crrna_emb = _get_batch_embeddings(crrna_spacers, cache, device)

            # Label noise on efficiency
            noise = torch.randn_like(activity) * 0.03
            activity_noisy = (activity + noise).clamp(0, 1)

            optimizer.zero_grad()

            output = model(target_onehot=target_oh, crrna_rnafm_emb=crrna_emb)

            # Efficiency loss
            l_eff = efficiency_loss_fn(output["efficiency"].squeeze(-1), activity_noisy)

            # Domain adversarial loss
            l_dom = domain_loss_fn(output["domain_logits"], domain_ids)

            # Total loss: efficiency + weighted domain (GRL handles sign reversal)
            loss = l_eff + args.lambda_domain * l_dom

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_eff_loss += l_eff.item()
            total_dom_loss += l_dom.item()

            # Domain classification accuracy
            dom_preds = output["domain_logits"].argmax(dim=-1)
            total_dom_correct += (dom_preds == domain_ids).sum().item()
            total_dom_count += len(domain_ids)
            n_batches += 1

        scheduler.step()

        avg_eff = total_eff_loss / max(n_batches, 1)
        avg_dom = total_dom_loss / max(n_batches, 1)
        dom_acc = total_dom_correct / max(total_dom_count, 1)
        chance = 1.0 / n_domains

        # --- Validation on Kim 2018 val set ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                target_oh = batch["target_onehot"].to(device)
                crrna_emb = _get_batch_embeddings(batch["crrna_spacer"], cache, device)
                output = model(target_onehot=target_oh, crrna_rnafm_emb=crrna_emb)
                all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
                all_labels.extend(batch["efficiency"].tolist())

        val_rho, _ = spearmanr(all_preds, all_labels)

        # Early stopping
        if val_rho > best_rho:
            best_rho = val_rho
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_rho": val_rho,
                "n_domains": n_domains,
                "grl_lambda": grl_lambda,
                "config": {
                    "n_domains": n_domains,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "easydesign": easydesign_available,
                },
            }, args.output)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "Epoch %3d | Eff: %.4f | Dom: %.4f | DomAcc: %.2f (chance=%.2f) | "
                "Val rho: %.4f | Best: %.4f | GRL: %.3f | LR: %.2e",
                epoch + 1, avg_eff, avg_dom, dom_acc, chance,
                val_rho, best_rho, grl_lambda, current_lr,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d. Best rho = %.4f", epoch + 1, best_rho)
            break

    # --- Load best and evaluate ---
    best_ckpt = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_epoch = best_ckpt["epoch"] + 1
    logger.info("Loaded best model from epoch %d (rho = %.4f)", best_epoch, best_ckpt["val_rho"])

    logger.info("=" * 60)
    logger.info("Evaluation")
    logger.info("=" * 60)

    def evaluate_loader(loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                target_oh = batch["target_onehot"].to(device)
                crrna_emb = _get_batch_embeddings(batch["crrna_spacer"], cache, device)
                output = model(target_onehot=target_oh, crrna_rnafm_emb=crrna_emb)
                all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
                all_labels.extend(batch["efficiency"].tolist())

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        rho, _ = spearmanr(preds, labels)
        r, _ = pearsonr(preds, labels)
        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)

        k = max(1, len(labels) // 5)
        top_pred_idx = np.argsort(preds)[-k:]
        top_true_idx = set(np.argsort(labels)[-k:])
        top_k_prec = len(set(top_pred_idx) & top_true_idx) / k

        return {
            "spearman_rho": round(float(rho), 4),
            "pearson_r": round(float(r), 4),
            "mse": round(float(mse), 4),
            "mae": round(float(mae), 4),
            "top_k_precision": round(float(top_k_prec), 4),
        }

    val_metrics = evaluate_loader(val_loader)
    test_metrics = evaluate_loader(test_loader)

    logger.info("Validation metrics:")
    for k, v in val_metrics.items():
        logger.info("  %-20s %.4f", k, v)
    logger.info("Test metrics (HT2 + HT3):")
    for k, v in test_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    # --- Summary ---
    config_str = f"Multi-dataset DA ({n_domains} domains"
    if easydesign_available:
        config_str += ", +EasyDesign"
    config_str += ")"

    print("\n" + "=" * 60)
    print(f"Compass-ML Row 6: {config_str}")
    print("=" * 60)
    print(f"Config:     {config_str}")
    print(f"Params:     {model.count_trainable_params():,}")
    print(f"Val rho:    {val_metrics['spearman_rho']:.4f}")
    print(f"Test rho:   {test_metrics['spearman_rho']:.4f}")
    print(f"Test r:     {test_metrics['pearson_r']:.4f}")
    print(f"Test MSE:   {test_metrics['mse']:.4f}")
    print(f"Top-20%%:   {test_metrics['top_k_precision']:.3f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val:   {best_ckpt['val_rho']:.4f}")
    print()
    print("Comparison to previous rows:")
    print(f"  CNN only:              test rho = 0.4959")
    print(f"  CNN + RNA-FM:          test rho = 0.5009")
    print(f"  CNN + RNA-FM + RLPA:   test rho = 0.5336")
    print(f"  + Multi-task:          test rho = 0.5361")
    print(f"  + LoRA + thermo:       test rho = 0.5373")
    print(f"  + Multi-dataset DA:    test rho = {test_metrics['spearman_rho']:.4f}")
    print("=" * 60)

    # Save results
    results = {
        "config": config_str,
        "params": model.count_trainable_params(),
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "n_domains": n_domains,
        "easydesign_used": easydesign_available,
        "seed": args.seed,
    }
    results_path = args.output.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
