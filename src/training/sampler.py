import json

import torch.utils.data as tdata

from src.constants import XBD_S12_PATH


def get_weighted_sampler(ds: tdata.Dataset) -> tdata.WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler for the given dataset based on precomputed sample weights.

    Each class is assigned a weight w_i inversely proportional to its frequency in the training set. Then each patch
    is assigned a weight based on the classes it contains.
    """

    # Read weights from file
    d_weights = get_training_weights(which_split=ds.which_split)

    # Map weights to dataset samples
    ds.meta["uid_weights"] = ds.meta["xbd_uid"].map(d_weights)  # add weights to dataset meta
    sample_weights = ds.meta["uid_weights"].to_list()

    return tdata.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def get_training_weights(which_split: str) -> dict:
    """Load or compute weights for each xbd_uid for the given split."""

    weights_fp = XBD_S12_PATH / "stats" / f"uid_weights_{which_split}_split.json"
    if not weights_fp.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_fp}. Please make sure they exist.")

    with open(weights_fp, "r") as f:
        weights = json.load(f)  # dict {xbd_uid: weights}

    return weights
