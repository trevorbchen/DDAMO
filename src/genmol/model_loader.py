"""Merge a DDPP-finetuned checkpoint into the base pretrained checkpoint.

The result is a standard Lightning checkpoint loadable by any Sampler(path=...).

Usage:
    from genmol.model_loader import merge_ddpp_checkpoint

    merged = merge_ddpp_checkpoint("model_v2.ckpt", "outputs/ddpp/ddpp_checkpoint.pt")
    sampler = Sampler(path=merged)
"""

import logging
import os
import tempfile

import torch

logger = logging.getLogger(__name__)


def merge_ddpp_checkpoint(base_path: str, ddpp_path: str) -> str:
    """Merge DDPP backbone weights into the base Lightning checkpoint.

    The DDPP checkpoint stores ``backbone_state_dict`` (EMA weights).
    We load the base checkpoint, overwrite the backbone keys, and save
    to a temporary file that ``GenMol.load_from_checkpoint`` can read.

    Returns:
        Path to the merged temporary checkpoint.
    """
    logger.info("Merging DDPP checkpoint %s into base %s", ddpp_path, base_path)

    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    ddpp_ckpt = torch.load(ddpp_path, map_location="cpu", weights_only=False)

    ddpp_backbone = ddpp_ckpt["backbone_state_dict"]

    state_dict = base_ckpt["state_dict"]
    for key, value in ddpp_backbone.items():
        prefixed = f"backbone.{key}" if not key.startswith("backbone.") else key
        if prefixed in state_dict:
            state_dict[prefixed] = value
        else:
            logger.warning("DDPP key %s not found in base checkpoint, skipping", prefixed)

    base_ckpt["state_dict"] = state_dict

    fd, merged_path = tempfile.mkstemp(suffix=".ckpt", prefix="ddpp_merged_")
    os.close(fd)
    torch.save(base_ckpt, merged_path)
    logger.info("Saved merged checkpoint → %s", merged_path)

    return merged_path


# backward-compat alias
_merge_ddpp_checkpoint = merge_ddpp_checkpoint
