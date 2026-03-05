"""Smoke tests for Compass-ML architecture.

Run from the compass/ root:
    python compass-net/test_compass_ml.py
"""

from __future__ import annotations

import sys
import os
import importlib
import importlib.util

# ---------------------------------------------------------------------------
# Bootstrap: make compass-net/ importable as "compass_ml" despite the hyphen
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # compass-net/
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)                   # compass/

def _register(full_name: str, file_path: str, search_paths: list[str] | None = None):
    """Register a module/package in sys.modules from an explicit file path."""
    spec = importlib.util.spec_from_file_location(
        full_name, file_path, submodule_search_locations=search_paths,
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = full_name if search_paths else full_name.rsplit(".", 1)[0]
    if search_paths:
        mod.__path__ = search_paths
    sys.modules[full_name] = mod
    return spec, mod

def _setup():
    """Register compass-net/ as the 'compass_ml' package."""
    # 1. Register the top-level package
    pkg_init = os.path.join(_SCRIPT_DIR, "__init__.py")
    _register("compass_ml", pkg_init, [_SCRIPT_DIR])

    # 2. Register each subpackage
    for sub in ["branches", "attention", "heads", "losses", "data",
                "training", "evaluation", "features"]:
        sub_dir = os.path.join(_SCRIPT_DIR, sub)
        sub_init = os.path.join(sub_dir, "__init__.py")
        if os.path.isfile(sub_init):
            _register(f"compass_ml.{sub}", sub_init, [sub_dir])

    # 3. Register compass_ml.compass_ml (the main model module)
    _register("compass_ml.compass_ml", os.path.join(_SCRIPT_DIR, "compass_ml.py"))

    # 4. Register individual child modules so relative imports resolve
    for sub in ["branches", "attention", "heads", "losses", "data",
                "training", "evaluation", "features"]:
        sub_dir = os.path.join(_SCRIPT_DIR, sub)
        if not os.path.isdir(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            if fname.endswith(".py") and fname != "__init__.py":
                mod_name = fname[:-3]
                full = f"compass_ml.{sub}.{mod_name}"
                fpath = os.path.join(sub_dir, fname)
                _register(full, fpath)

    # 5. Execute modules in dependency order
    _exec("compass_ml.branches")
    _exec("compass_ml.attention")
    _exec("compass_ml.heads")
    _exec("compass_ml.losses")
    _exec("compass_ml.data")
    _exec("compass_ml.features")
    _exec("compass_ml.compass_ml")
    _exec("compass_ml.evaluation")
    _exec("compass_ml.training")
    _exec("compass_ml")

def _exec(full_name: str):
    """Execute a registered module and all its children."""
    mod = sys.modules.get(full_name)
    if mod is None:
        return
    spec = mod.__spec__
    if spec is None or spec.loader is None:
        return

    # Execute child .py modules first if this is a package
    prefix = full_name + "."
    children = [k for k in sys.modules if k.startswith(prefix) and k.count(".") == full_name.count(".") + 1 and not k.endswith(".__init__")]
    for child_name in sorted(children):
        child = sys.modules[child_name]
        if child.__spec__ and child.__spec__.loader:
            try:
                child.__spec__.loader.exec_module(child)
            except Exception:
                pass  # optional deps

    # Then execute the package itself
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import torch


def test_cnn_branch():
    from compass_ml.branches.cnn_branch import CNNBranch
    model = CNNBranch(in_channels=4, branches=32, out_dim=64)
    x = torch.randn(2, 4, 34)
    out = model(x)
    assert out.shape == (2, 34, 64), f"Expected (2, 34, 64), got {out.shape}"
    print(f"  CNN branch: {sum(p.numel() for p in model.parameters())} params")


def test_rnafm_branch():
    from compass_ml.branches.rnafm_branch import RNAFMBranch
    model = RNAFMBranch(embed_dim=640, proj_dim=64, target_len=34)
    emb = torch.randn(2, 20, 640)
    out = model(emb)
    assert out.shape == (2, 34, 64), f"Expected (2, 34, 64), got {out.shape}"
    assert out[:, :4, :].abs().sum() == 0, "PAM positions should be zero"
    assert out[:, 24:, :].abs().sum() == 0, "Flanking positions should be zero"
    assert out[:, 4:24, :].abs().sum() > 0, "Spacer positions should be non-zero"
    print(f"  RNA-FM branch: {sum(p.numel() for p in model.parameters())} params")
    print("  20->34 alignment: PAM=zeros, spacer=projected, flanking=zeros")


def test_rloop_attention():
    from compass_ml.attention.rloop_attention import RLoopAttention
    model = RLoopAttention(d_model=128)
    model.eval()  # disable dropout so attn weights sum to 1
    x = torch.randn(2, 34, 128)
    out, attn = model(x)
    assert out.shape == (2, 34, 128), f"Expected (2, 34, 128), got {out.shape}"
    assert attn.shape == (2, 34, 34), f"Expected (2, 34, 34), got {attn.shape}"
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Attn should sum to 1"
    print(f"  RLPA: {sum(p.numel() for p in model.parameters())} params")


def test_compass_ml_cnn_only():
    from compass_ml.compass_ml import CompassML
    model = CompassML(use_rnafm=False, use_rloop_attention=False, multitask=False)
    x = torch.randn(2, 4, 34)
    out = model(target_onehot=x)
    assert "efficiency" in out
    assert out["efficiency"].shape == (2, 1)
    assert "discrimination" not in out
    n = model.count_trainable_params()
    print(f"  CompassML (CNN only): {n} params")


def test_compass_ml_dual_branch():
    from compass_ml.compass_ml import CompassML
    model = CompassML(use_rnafm=True, use_rloop_attention=False, multitask=False)
    x = torch.randn(2, 4, 34)
    emb = torch.randn(2, 20, 640)
    out = model(target_onehot=x, crrna_rnafm_emb=emb)
    assert out["efficiency"].shape == (2, 1)
    n = model.count_trainable_params()
    print(f"  CompassML (CNN + RNA-FM): {n} params")


def test_compass_ml_full():
    from compass_ml.compass_ml import CompassML
    model = CompassML(use_rnafm=True, use_rloop_attention=True, multitask=True)
    x = torch.randn(2, 4, 34)
    emb = torch.randn(2, 20, 640)
    wt = torch.randn(2, 4, 34)
    out = model(target_onehot=x, crrna_rnafm_emb=emb, wt_target_onehot=wt)
    assert out["efficiency"].shape == (2, 1)
    assert "discrimination" in out and out["discrimination"].shape == (2, 1)
    assert "attn_weights" in out and out["attn_weights"].shape == (2, 34, 34)
    n = model.count_trainable_params()
    print(f"  CompassML (full): {n} params")
    assert n < 250_000, f"Too many params: {n} > 250K"


def test_compass_ml_with_scalars():
    from compass_ml.compass_ml import CompassML
    model = CompassML(use_rnafm=True, n_scalar_features=3)
    x = torch.randn(2, 4, 34)
    emb = torch.randn(2, 20, 640)
    scalars = torch.randn(2, 3)
    out = model(target_onehot=x, crrna_rnafm_emb=emb, scalar_features=scalars)
    assert out["efficiency"].shape == (2, 1)
    print(f"  CompassML (+ 3 scalars): {model.count_trainable_params()} params")


def test_multitask_loss():
    # Re-exec in case bootstrap didn't fully resolve
    mod = sys.modules.get("compass_ml.losses.multitask_loss")
    if mod and not hasattr(mod, "MultiTaskLoss"):
        mod.__spec__.loader.exec_module(mod)
    from compass_ml.losses.multitask_loss import MultiTaskLoss
    loss_fn = MultiTaskLoss(lambda_disc=0.3, lambda_rank=0.5)
    pred_eff = torch.rand(8, 1)
    true_eff = torch.rand(8)
    pred_disc = torch.rand(8, 1) * 5
    true_disc = torch.rand(8) * 5

    losses = loss_fn(pred_eff, true_eff)
    assert "total" in losses and "efficiency" in losses
    assert "discrimination" not in losses

    losses = loss_fn(pred_eff, true_eff, pred_disc, true_disc)
    assert "discrimination" in losses
    print(f"  MultiTaskLoss: total={losses['total'].item():.4f}")


def test_spearman_loss():
    from compass_ml.losses.spearman_loss import differentiable_spearman
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    rho = differentiable_spearman(x, y)
    assert rho.item() > 0.95, f"Expected rho > 0.95, got {rho.item():.4f}"
    rho.backward()
    assert x.grad is not None
    print(f"  Differentiable Spearman: rho={rho.item():.4f}")


def test_paired_loader():
    from compass_ml.data.paired_loader import PairedTargetDataset
    seqs = ["TTTGATCGATCGATCGATCGATCGATCGATCGAT"] * 5
    acts = [0.8, 0.6, 0.9, 0.3, 0.7]
    ds = PairedTargetDataset(seqs, acts)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["mutant_target_onehot"].shape == (4, 34)
    assert sample["wildtype_target_onehot"].shape == (4, 34)
    assert isinstance(sample["crrna_spacer"], str)
    assert "T" not in sample["crrna_spacer"], "crRNA should use U, not T"
    print(f"  PairedTargetDataset: {len(ds)} pairs from 5 sequences")


def test_embedding_cache():
    import tempfile
    from compass_ml.data.embedding_cache import EmbeddingCache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(tmpdir)
        seqs = ["AUGCCGAUUCGA", "GCUAUUGCCAAU"]
        embs = [torch.randn(12, 640), torch.randn(12, 640)]
        cache.put_batch(seqs, embs)
        assert cache.has("AUGCCGAUUCGA")
        assert not cache.has("NONEXISTENT")
        retrieved = cache.get("AUGCCGAUUCGA")
        assert retrieved is not None and retrieved.shape == (12, 640)
        assert torch.allclose(retrieved, embs[0])
    print("  EmbeddingCache: put/get round-trip OK")


def test_clustered_split():
    from compass_ml.data.split_strategy import clustered_split
    import random
    random.seed(42)
    seqs = ["".join(random.choices("ACGT", k=34)) for _ in range(100)]
    train, val, test = clustered_split(seqs, identity_threshold=0.80)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == 100
    print(f"  Clustered split: train={len(train)}, val={len(val)}, test={len(test)}")


def test_discrimination_head():
    from compass_ml.heads.discrimination_head import DiscriminationHead
    head = DiscriminationHead(input_dim=64, hidden_dim=64)
    mut = torch.randn(2, 64)
    wt = torch.randn(2, 64)
    out = head(mut, wt)
    assert out.shape == (2, 1)
    assert (out > 0).all(), "Softplus output should be positive"
    print(f"  DiscriminationHead: {sum(p.numel() for p in head.parameters())} params")


def test_calibration():
    import numpy as np
    from compass_ml.evaluation.calibration import reliability_diagram, check_calibration
    np.random.seed(42)
    preds = np.random.rand(100)
    targets = np.clip(preds + np.random.randn(100) * 0.1, 0, 1)
    result = reliability_diagram(preds, targets)
    assert "ece" in result and "bins" in result
    is_cal, ece = check_calibration(preds, targets)
    print(f"  Calibration: ECE={ece:.4f}, calibrated={is_cal}")


def test_temperature_scaling():
    from compass_ml.evaluation.calibration import TemperatureScaling
    ts = TemperatureScaling()
    logits = torch.randn(50)
    targets = torch.sigmoid(logits + torch.randn(50) * 0.5)
    temp = ts.fit(logits, targets)
    print(f"  Temperature scaling: fitted T={temp:.4f}")


def main():
    _setup()

    print("=" * 60)
    print("Compass-ML Architecture Tests")
    print("=" * 60)

    tests = [
        ("CNN branch", test_cnn_branch),
        ("RNA-FM branch (20->34 alignment)", test_rnafm_branch),
        ("R-Loop Propagation Attention", test_rloop_attention),
        ("CompassML: CNN only", test_compass_ml_cnn_only),
        ("CompassML: CNN + RNA-FM", test_compass_ml_dual_branch),
        ("CompassML: full (CNN+RNA-FM+RLPA+MT)", test_compass_ml_full),
        ("CompassML: with scalar features", test_compass_ml_with_scalars),
        ("Discrimination head", test_discrimination_head),
        ("Differentiable Spearman", test_spearman_loss),
        ("Multi-task loss", test_multitask_loss),
        ("Paired target loader", test_paired_loader),
        ("Embedding cache", test_embedding_cache),
        ("Clustered split", test_clustered_split),
        ("Calibration / reliability diagram", test_calibration),
        ("Temperature scaling", test_temperature_scaling),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            print("  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
