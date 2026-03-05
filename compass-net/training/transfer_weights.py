"""Transfer CNN weights from SeqCNN v1 to Compass-ML CNN branch.

The CNN branch is architecturally IDENTICAL to v1's conv layers,
with matching layer names for direct weight transfer:

    SeqCNN v1 (compass/scoring/seq_cnn.py)    Compass-ML CNNBranch
    ---------------------------------------- ----------------------------
    multi_scale.branch3.0 (Conv1d)          cnn.branch3.0 (Conv1d)
    multi_scale.branch3.1 (BatchNorm1d)     cnn.branch3.1 (BatchNorm1d)
    multi_scale.branch5.*                   cnn.branch5.*
    multi_scale.branch7.*                   cnn.branch7.*
    dilated.conv1.*                         cnn.dilated1.*
    dilated.conv2.*                         cnn.dilated2.*
    reduce.*                                cnn.reduce.*

v1 head (pool + dense layers) is NOT transferred -- Compass-ML has
its own efficiency head with different input dimensions.

Usage:
    from compass_ml.training.transfer_weights import transfer_v1_to_compass_ml
    model = CompassML(use_rnafm=True)
    model, n = transfer_v1_to_compass_ml("compass/weights/seq_cnn_best.pt", model)
    print(f"Transferred {n} parameter tensors")
"""

from __future__ import annotations

import logging

import torch

from ..compass_ml import CompassML

logger = logging.getLogger(__name__)

# Mapping: v1 layer prefix -> Compass-ML layer prefix
LAYER_MAPPING = {
    "multi_scale.branch3": "cnn.branch3",
    "multi_scale.branch5": "cnn.branch5",
    "multi_scale.branch7": "cnn.branch7",
    "dilated.conv1": "cnn.dilated1",
    "dilated.conv2": "cnn.dilated2",
    "reduce": "cnn.reduce",
}

# v1 layers that should NOT be transferred (head belongs to v1 only)
SKIP_PREFIXES = {"pool", "head"}


def transfer_v1_to_compass_ml(
    v1_checkpoint: str,
    compass_ml: CompassML,
    strict_cnn: bool = True,
) -> tuple[CompassML, int]:
    """Load v1 CNN weights into Compass-ML's CNN branch.

    Args:
        v1_checkpoint: path to v1 checkpoint (.pt file).
        compass_ml: CompassML instance with randomly initialised CNN branch.
        strict_cnn: if True, raise on shape mismatch in mapped layers.

    Returns:
        (model, n_transferred) -- model with loaded weights,
        count of parameter tensors transferred.
    """
    v1_state = torch.load(v1_checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in v1_state:
        v1_state = v1_state["model_state_dict"]

    compass_state = compass_ml.state_dict()
    transferred = 0
    skipped = 0

    for v1_name, v1_param in v1_state.items():
        # Skip v1 head layers
        if any(v1_name.startswith(p) for p in SKIP_PREFIXES):
            skipped += 1
            continue

        # Map v1 name to Compass-ML name
        compass_name = None
        for v1_prefix, compass_prefix in LAYER_MAPPING.items():
            if v1_name.startswith(v1_prefix):
                compass_name = v1_name.replace(v1_prefix, compass_prefix, 1)
                break

        if compass_name is None:
            logger.debug("No mapping for v1 layer: %s", v1_name)
            skipped += 1
            continue

        if compass_name not in compass_state:
            logger.warning("Mapped name %s not found in Compass-ML", compass_name)
            skipped += 1
            continue

        if compass_state[compass_name].shape != v1_param.shape:
            msg = (
                f"Shape mismatch: {v1_name} {tuple(v1_param.shape)} "
                f"vs {compass_name} {tuple(compass_state[compass_name].shape)}"
            )
            if strict_cnn:
                raise ValueError(msg)
            logger.warning(msg)
            skipped += 1
            continue

        compass_state[compass_name] = v1_param
        transferred += 1

    compass_ml.load_state_dict(compass_state)

    # Report
    non_cnn_params = sum(
        1 for n in compass_state if not n.startswith("cnn.")
    )
    logger.info(
        "Weight transfer: %d tensors transferred, %d skipped, "
        "%d non-CNN layers randomly initialised",
        transferred, skipped, non_cnn_params,
    )

    return compass_ml, transferred
